import torch
import torch.nn as nn
import torch.nn.functional as F


class MNISTModel(nn.Module):
    def __init__(self):
        super(MNISTModel, self).__init__()

        # 第一部分：卷积层 + 激活 + 池化
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3)  # 输入通道数=1，输出通道数=32
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3)  # 输入通道数=32，输出通道数=32
        self.pool1 = nn.MaxPool2d(kernel_size=2)  # 最大池化层

        # 第二部分：卷积层 + 激活 + 池化
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3)  # 输入通道数=32，输出通道数=64
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3)  # 输入通道数=64，输出通道数=64
        self.pool2 = nn.MaxPool2d(kernel_size=2)  # 最大池化层

        # 全连接层
        self.flatten_size = 64 * 4 * 4  # 展平后的特征维度（根据输入尺寸28x28计算）
        self.fc1 = nn.Linear(self.flatten_size, 200)  # 第一个全连接层
        self.fc2 = nn.Linear(200, 200)  # 第二个全连接层
        self.fc3 = nn.Linear(200, 10)  # 输出层（10分类）

    def forward(self, x):
        # 第一部分：卷积 + 激活 + 池化
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool1(x)

        # 第二部分：卷积 + 激活 + 池化
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = self.pool2(x)

        # 展平
        x = x.view(-1, self.flatten_size)

        # 全连接层
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)  # 最后一层一般不需要激活函数（在交叉熵损失中会自动计算Softmax）

        return x

# 加载预训练权重
def load_weights(model, weight_path):
    model.load_state_dict(torch.load(weight_path))
    return model
