import os
import shutil
import sys
import matplotlib.pyplot as plt
import torch
import torchvision
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from torch.utils.data import DataLoader

# 定义训练设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 网络、损失函数，数据
# 加载数据集
train_dataset = torchvision.datasets.MNIST('../data/MNIST', train=True,
                                             transform=torchvision.transforms.ToTensor(), download=True)
test_dataset = torchvision.datasets.MNIST('../data/MNIST', train=False,
                                            transform=torchvision.transforms.ToTensor(), download=True)
# 打包
train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=64, shuffle=True)


# 构建网络
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.pool3 = nn.MaxPool2d(kernel_size=2)
        self.relu = nn.ReLU()
        self.flt = nn.Flatten()
        self.fc1 = nn.Linear(in_features=128 * 3 * 3, out_features=625)
        self.fc2 = nn.Linear(in_features=625, out_features=10)

    def forward(self, x):
        x = self.pool1(self.relu(self.conv1(x)))
        x = self.pool2(self.relu(self.conv2(x)))
        x = self.pool3(self.relu(self.conv3(x)))
        x = self.flt(x)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# 实例化
net = Net()
net = net.to(device)  # 将网络放到GPU运算
# print(net)
# 定义损失函数
loss_fn = nn.CrossEntropyLoss()  # 交叉熵
loss_fn = loss_fn.to(device)
# 定义优化器、学习率
learning_rate = 1e-2
optimizer = torch.optim.SGD(params=net.parameters(), lr=learning_rate)  # lr是学习率
# 定义超参数（训练所需要的一些参数）
total_train_step = 0  # 记录训练次数
total_test_step = 0  # 记录测试（验证）次数
running_loss = 0  # 记录整体的loss
epochs = 20  # 训练轮数
accuracy = []  # 用来记录准确率画图用

# 添加tensorboard用于可视化
i = 1  # 日志文件夹的序号
logs_path = "logs_mnist/logs_mnist_{}".format(i)  # 日志文件的保存路径，第一个为logs_mnist_1
while os.path.exists(logs_path):  # 判断目录中是否有这个文件夹
    i += 1  # 如果有的话序号加1
    logs_path = "logs_mnist/logs_mnist_{}".format(i)
writer = SummaryWriter(logs_path)

for epoch in range(epochs):  # 训练轮数
    # print("-----第{}轮训练开始-----".format(epoch))
    net.train()  # 训练模式
    train_loss_time = 0
    train_dataloader = tqdm(train_dataloader, desc="train", file=sys.stdout, colour="Green")  # 显示进度条
    for data in train_dataloader:  # 取数据
        imgs, targets = data  # 图像和标签分别取出
        imgs, targets = imgs.to(device), targets.to(device)
        opts = net(imgs)  # 将图像送入网络，得到预测值opts
        loss = loss_fn(opts, targets)  # 损失函数计算损失loss
        # 优化器优化模型
        optimizer.zero_grad()  # 梯度清零
        loss.backward()  # 反向传播，得到每一个参数点的梯度
        optimizer.step()  # 优化，一次训练结束
        total_train_step += 1  # 记录训练次数
        running_loss += loss.item()  # 记录整体loss
        train_loss_time += 1  # 记录计算loss的次数
        # if total_train_step % 100 == 0:
        #     print("训练次数:{}，loss:{}".format(total_train_step, loss.item()))  # 输出每一轮的loss
    print("epoch:{}  loss:{:.3f}".format(epoch + 1, running_loss / train_loss_time))  # 输出loss
    # 测试步骤开始
    net.eval()  # 测试模式
    total_test_loss = 0  # 记录测试loss
    total_targets_num = 0  # 记录标签数量
    total_acc = 0  # 记录整体上的准确的数量
    test_loss_time = 0
    with torch.no_grad():  # 不进行梯度运算，梯度运算是为了训练模型，测试时不需要
        test_dataloader = tqdm(test_dataloader, desc="test ", file=sys.stdout, colour="red")  # 显示进度条
        for data in test_dataloader:  # 取数据
            imgs, targets = data  # 图像和标签分别取出
            imgs, targets = imgs.to(device), targets.to(device)
            opts = net(imgs)  # 将图像送入训练好的网络，得到预测值opts
            loss = loss_fn(opts, targets)  # 损失函数计算损失loss
            total_test_loss += loss  # 记录测试loss
            test_loss_time += 1
            _, pre = torch.max(opts.data, dim=1)  # 沿着第一维度，找最大值的下标，返回最大值和下标
            total_targets_num += targets.size(0)  # 因为batch_size定义的是64,所以targets.size(0)每个都是64个元素，得到总的数量
            total_acc += (pre == targets).sum().item()  # 记录整体上的准确的数量，pre == targets判断是否正确，正确是1，不正确是0
    accuracy.append(100 * (total_acc / total_targets_num))  # 用来记录准确率画图用
    print("Accuracy on test set:{:.2f}%".format(100 * (total_acc / total_targets_num)))
    total_test_step += 1
    writer.add_scalar("test_loss", total_test_loss / test_loss_time, total_test_step)
    writer.add_scalar("test_accuracy", 100 * (total_acc / total_targets_num), total_test_step)
writer.close()
# 用matplotlib画图
plt.figure(figsize=(8, 6))  # 定义图及大小
plt.title("MNIST")  # 标题
plt.xlabel("epoch")  # x坐标轴
plt.ylabel("accuracy")  # y坐标轴
plt.grid(visible=True)  # 显示网格
plt.plot(range(epochs), accuracy)  # 画图
plt.show()  # 展示
