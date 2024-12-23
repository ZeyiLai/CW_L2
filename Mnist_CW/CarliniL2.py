import numpy as np
import json
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as Data
import torchvision.utils as vutils
from torchvision import models
import torchvision.datasets as dsets
from PIL import Image
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import os
from tqdm import tqdm

class CWL2Attack:
    def __init__(self, model, device, targeted=False, c=1e-3, kappa=0, max_iter=1000, learning_rate=0.01):
        """
        初始化CW L2攻击类
        :param model: 待攻击的模型
        :param device: 计算设备（如 'cuda' 或 'cpu'）
        :param targeted: 是否为定向攻击（默认为False）
        :param c: 损失平衡系数
        :param kappa: 信心参数
        :param max_iter: 最大迭代次数
        :param learning_rate: 优化器的学习率
        """
        self.model = model
        self.device = device
        self.targeted = targeted
        self.c = c
        self.kappa = kappa
        self.max_iter = max_iter
        self.learning_rate = learning_rate

    def f(self, x, labels):
        """
        CW L2攻击中的 f 函数
        :param x: 输入样本
        :param labels: 标签（targeted模式下为目标类标签，untargeted模式下为原始标签）
        :return: 损失值
        """
        outputs = self.model(x)  # 获取模型输出
        one_hot_labels = torch.eye(len(outputs[0]), device=self.device)[labels].to(self.device)  # 构造one-hot标签
        other = torch.max((1 - one_hot_labels) * outputs, dim=1)[0]  # 取非目标类的最大输出
        real = torch.max(one_hot_labels * outputs, dim=1)[0]  # 取目标类的输出

        if self.targeted:
            # 定向攻击：优化目标类的输出大于非目标类
            return torch.clamp((other - real), min=-self.c)
        else:
            # 非定向攻击：优化使得原始类的输出低于其他类
            return torch.clamp((real - other), min=-self.c)

    def attack(self, images, labels, use_tqdm=True):
        """
        执行CW L2攻击
        :param images: 输入图像
        :param labels: 对应标签
        :param use_tqdm: 是否使用进度条显示
        :return: 对抗样本
        """
        images = images.to(self.device)  # 将输入图像转移到指定设备
        w = torch.zeros_like(images, requires_grad=True).to(self.device)  # 初始化扰动变量 w

        optimizer = optim.Adam([w], lr=self.learning_rate)  # 使用Adam优化器优化 w
        prev = 1e10  # 初始化前一轮的损失值为一个较大数值

        iteration_range = range(self.max_iter)  # 设置迭代范围

        # 如果启用进度条，使用 tqdm 包裹迭代范围
        if use_tqdm:
            iteration_range = tqdm(iteration_range, desc="Attacking", ncols=100)

        # 迭代优化
        for step in iteration_range:
            # 将 w 转换到像素空间（通过 tanh 确保值在 [0,1] 范围内）
            a = 1 / 2 * (nn.Tanh()(w) + 1)

            # 计算MSE损失（与原始图像的差异）
            loss1 = nn.MSELoss(reduction='sum')(a, images)

            # 计算分类损失
            loss2 = self.f(a, labels).sum()

            # 总损失
            cost = loss1 + loss2

            # 优化步骤
            optimizer.zero_grad()
            cost.backward()
            optimizer.step()

            # 打印每步的损失信息（可选）
            # print(f"Step {step + 1}/{self.max_iter}, Cost: {cost.item():.4f}, Loss1: {loss1.item():.4f}, Loss2: {loss2.item():.4f}")

            # 如果损失在若干步内不收敛，则提前停止
            if step % (self.max_iter // 10) == 0:
                if cost > prev:
                    print('攻击提前停止，损失收敛...')
                    return a  # 返回当前生成的对抗样本
                prev = cost

        # 最终返回生成的对抗样本
        attack_images = 1 / 2 * (nn.Tanh()(w) + 1)
        return attack_images
