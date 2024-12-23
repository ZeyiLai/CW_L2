import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset
from model import *
from tqdm import tqdm

class MNISTDataset:
    def __init__(self):
        """
        初始化MNIST数据集的加载和预处理
        """
        # 数据预处理（将图像转换为张量）
        self.transform = transforms.Compose([
            transforms.ToTensor(),
        ])

        # 数据集加载
        self.train_dataset = datasets.MNIST(root="./data", train=True, download=True, transform=self.transform)
        self.test_dataset = datasets.MNIST(root="./data", train=False, download=True, transform=self.transform)

    def get_test_labels(self):
        # 获取测试集的标签
        return self.test_dataset.targets  # 返回标签张量

    def get_train_labels(self):
        # 获取测试集的标签
        return self.train_dataset.targets  # 返回标签张量

    def get_train_data(self):
        # 获取训练集的数据
        return self.train_dataset

    def get_test_data(self):
        # 获取测试集的数据
        return self.test_dataset



class MNISTTrainer:
    def __init__(self, model, device="cpu", batch_size=64, lr=0.001):
        """
        初始化训练器
        :param model: 要训练的模型
        :param device: 计算设备，默认为'cpu'
        :param batch_size: 批量大小
        :param lr: 学习率
        """
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)

        # 数据预处理
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))  # 归一化到 [-1, 1]
        ])

        # 数据集加载
        self.train_dataset = datasets.MNIST(root="./data", train=True, download=True, transform=self.transform)
        self.test_dataset = datasets.MNIST(root="./data", train=False, download=True, transform=self.transform)

        self.train_loader = DataLoader(self.train_dataset, batch_size=batch_size, shuffle=True)
        self.test_loader = DataLoader(self.test_dataset, batch_size=batch_size, shuffle=False)

        # 损失函数和优化器
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)

        #存储训练损失
        self.loss_history = []

    def train(self, epochs=10):
        """
        模型训练
        :param epochs: 训练轮数
        :return: 训练后的模型
        """

        self.model.train()

        # 训练损失记录
        loss_history = [] # 用于记录每个epoch的损失

        for epoch in range(epochs):
            total_loss = 0

            # 使用 tqdm 包装 DataLoader，显示进度条
            progress_bar = tqdm(self.train_loader, desc=f"Epoch {epoch + 1}/{epochs}", dynamic_ncols=True)

            for batch_idx, (data, target) in enumerate(progress_bar):
                # 将数据和标签转移到计算设备
                data, target = data.to(self.device), target.to(self.device)

                # 前向传播
                outputs = self.model(data)
                loss = self.criterion(outputs, target)

                # 反向传播
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()

                # 更新进度条显示损失
                progress_bar.set_postfix(loss=total_loss / (batch_idx + 1))

            # 记录本 epoch 的平均损失,并添加到损失历史中
            avg_loss = total_loss / len(self.train_loader)
            loss_history.append(avg_loss)
        self.loss_history = loss_history # 保存损失历史

        return self.model

    def evaluate(self):
        """
        模型评估，计算测试集准确率
        :return: 准确率百分比
        """
        self.model.eval()
        correct = 0
        total = 0
        with torch.no_grad():  # 关闭梯度计算
            for data, target in self.test_loader:
                data, target = data.to(self.device), target.to(self.device)
                outputs = self.model(data) # 前向传播
                _, predicted = torch.max(outputs, 1) # 获取预测结果
                total += target.size(0) # 总样本数
                correct += (predicted == target).sum().item() # 正确预测数

        accuracy = 100 * correct / total
        print(f"Test Accuracy: {accuracy:.2f}%")
        return accuracy


    def print_train_loss(self):
        """
        打印每个训练轮次的损失
        """

        epoch = 0
        epochs = len(self.loss_history)
        for avg_loss in self.loss_history:
            print(f"Epoch [{epoch + 1}/{epochs}], Loss: {avg_loss:.4f}")
            epoch = epoch + 1

    def save_model(self, path):
        """
        保存整个模型
        :param path: 模型保存路径
        """
        torch.save(self.model, path)

    def save_weights(self, path):
        """
        保存模型权重
        :param path: 权重保存路径
        """
        torch.save(self.model.state_dict(), path)

def main():
    """
    主函数，用于训练和评估模型
    """
    save_path = "./models/mnist.pth" # 模型保存路径
    model = MNISTModel() # 初始化模型
    trainer = MNISTTrainer(model, device="cuda", batch_size=1024, lr=0.001) # 初始化训练器
    trainer.train(epochs=50) # 训练模型
    trainer.save_model(save_path) # 保存训练好的模型
    trainer.evaluate() # 评估模型性能

if __name__ == "__main__":
    main()
# Epoch 1/10: 100%|██████████| 469/469 [00:04<00:00, 101.67it/s, loss=0.219]
# Epoch 2/10: 100%|██████████| 469/469 [00:04<00:00, 97.88it/s, loss=0.0503]
# Epoch 3/10: 100%|██████████| 469/469 [00:06<00:00, 67.32it/s, loss=0.0342]
# Epoch 4/10: 100%|██████████| 469/469 [00:07<00:00, 65.78it/s, loss=0.0267]
# Epoch 5/10: 100%|██████████| 469/469 [00:05<00:00, 93.25it/s, loss=0.021]
# Epoch 6/10: 100%|██████████| 469/469 [00:05<00:00, 80.43it/s, loss=0.0162]
# Epoch 7/10: 100%|██████████| 469/469 [00:05<00:00, 80.85it/s, loss=0.0143]
# Epoch 8/10: 100%|██████████| 469/469 [00:07<00:00, 60.27it/s, loss=0.0121]
# Epoch 9/10: 100%|██████████| 469/469 [00:06<00:00, 77.78it/s, loss=0.0114]
# Epoch 10/10: 100%|██████████| 469/469 [00:05<00:00, 82.46it/s, loss=0.00897]
# Test Accuracy: 99.15%