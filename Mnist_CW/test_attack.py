import numpy as np
import torch
from model import load_weights
import matplotlib.pyplot as plt

from model import MNISTModel
from train import MNISTDataset
from CarliniL2 import *

def generate_data(data, samples, targeted=True, start=0):
    """
    生成输入数据以供攻击算法使用。

    :param data: 包含待攻击图像和标签的数据集
    :param samples: 生成的攻击样本数
    :param targeted: 如果为 True，则构造目标攻击样本，否则构造非目标攻击样本
    :param start: 从数据集中的哪个位置开始采样
    :return: 输入图像列表、目标标签列表、真实标签列表
    """
    # 初始化空列表，分别用于存储生成的输入图像和目标标签
    test_datas = data.get_test_data()
    inputs = []  # 用于存储输入图像
    targets = []  # 用于存储目标标签
    ground_truth = []  # 用于存储真实标签

    for i in range(samples):
        if targeted:
            # 定向攻击：目标标签为真实标签 +1 (取模确保在0-9范围内)
            inputs.append(test_datas[i + start][0])
            targets.append((test_datas[i + start][1] + 1) % 10)
            ground_truth.append(test_datas[i + start][1])
        else:
            # 非定向攻击：目标标签为真实标签
            inputs.append(test_datas[i + start][0])
            targets.append(test_datas[i + start][1])
            ground_truth.append(test_datas[i + start][1])
    # 返回生成的输入数据（图像）和目标标签
    return inputs, targets, ground_truth

def predict(model, tensor):
    """
    使用模型预测输入图像的类别。

    :param model: 用于预测的模型
    :param tensor: 输入的图像张量
    :return: 模型预测的类别索引
    """
    model.eval()
    with torch.no_grad():
        logits = model(tensor).squeeze().detach().cpu().numpy()  # 获取模型输出并转为NumPy数组
        max_index = np.argmax(logits)  # 获取概率最大的类别索引
    return max_index

def show(imgs, pert_imgs, ground_truth, predicts ,device):
    """
    显示原始图像和对抗样本，并比较其预测结果。

    :param imgs: 原始图像列表
    :param pert_imgs: 对抗样本列表
    :param ground_truth: 原始图像的真实标签
    :param predicts: 对抗样本的预测标签
    """
    modified = []
    for idx in range(len(imgs)):
        modified.append(imgs[idx].to(device) - pert_imgs[idx].to(device))

    # 设置图形大小和显示方式
    fig, axes = plt.subplots(len(imgs), 3, figsize=(12, len(imgs) * 4))  # 每行3张图片

    # 确保 axes 是一个二维数组
    if len(imgs) == 1:  # 如果只有一行
        axes = np.expand_dims(axes, axis=0)  # 添加维度，使其成为二维数组

    # 调整子图间距
    fig.subplots_adjust(hspace=0.6, wspace=0.5)  # 调整子图之间的间距

    for idx in range(len(imgs)):
        row = idx

        # 显示原始图像
        axes[row, 0].imshow(imgs[idx].squeeze().detach().cpu().numpy(), cmap='gray', vmin=0, vmax=1)
        axes[row, 0].set_title(f"\nGround Truth: {ground_truth[idx]}")
        axes[row, 0].axis('off')  # 不显示坐标轴

        # 显示扰动后的图像
        axes[row, 1].imshow(pert_imgs[idx].squeeze().detach().cpu().numpy(), cmap='gray', vmin=0, vmax=1)
        axes[row, 1].set_title(f"\nPrediction: {predicts[idx]}")
        axes[row, 1].axis('off')  # 不显示坐标轴

        # 显示噪点图像
        axes[row, 2].imshow(modified[idx].squeeze().detach().cpu().numpy(), cmap='gray', vmin=0, vmax=1)
        axes[row, 2].set_title(f"Perturbation")
        axes[row, 2].axis('off')  # 不显示坐标轴

    plt.tight_layout()  # 自动调整布局避免重叠
    plt.show()

def attacking(model, data, iters, device):
    """
    执行对抗攻击并收集成功攻击的样本。

    :param model: 待攻击的模型
    :param data: 数据集
    :return: 原始图像、对抗样本、真实标签和预测标签
    """
    showed = 0       # 记录已显示的成功攻击次数
    inputs, targets, ground_truth = generate_data(data, len(data.get_test_data()))
    imgs = []
    pert_imgs = []
    ground = []
    predicts = []

    # 遍历数据集，执行对抗攻击
    for idx in tqdm(range(iters), desc="Getting Ready", ncols=100):

        if(predict(model, inputs[idx].to(device)) != ground_truth[idx]):
            continue
        # 初始化CW L2攻击器
        attacker = CWL2Attack(model, device, False, 1, 0, 10, 0.01)

        # 对当前输入图像执行攻击
        adv = attacker.attack(inputs[idx], ground_truth[idx], False)

        result = predict(model, adv)

        if(result != ground_truth[idx]):
            imgs.append(inputs[idx])
            pert_imgs.append(adv)
            ground.append(ground_truth[idx])
            predicts.append(result)
        else:
            continue

    return imgs, pert_imgs, ground, predicts

def main():
    """
    主函数，用于加载数据、模型并执行CW L2攻击。
    """
    data = MNISTDataset()  # 加载MNIST数据集
    # model = torch.load('./models/mnist.pth')  # 加载预训练模型
    # torch.save(model.state_dict(), './models/mnist_weight.pth')

    model = load_weights(MNISTModel(), './models/mnist_weight.pth')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    model.to(device)
    imgs, pert_imgs, ground_truth, predicts = attacking(model, data, 5, device)  # 执行攻击
    show(imgs, pert_imgs, ground_truth, predicts, device)  # 显示结果

if __name__ == '__main__':
    main()
