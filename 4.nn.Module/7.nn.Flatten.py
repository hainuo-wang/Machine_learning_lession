import torchvision
from torch import nn
from torch.nn import Flatten
from torch.utils.data import DataLoader

# 加载
train_dataset = torchvision.datasets.CIFAR10('../data/CIFAR10', train=True,
                                             transform=torchvision.transforms.ToTensor(), download=True)
test_dataset = torchvision.datasets.CIFAR10('../data/CIFAR10', train=False,
                                            transform=torchvision.transforms.ToTensor(), download=True)
# 打包
train_loader = DataLoader(train_dataset, batch_size=64)
val_loader = DataLoader(test_dataset, batch_size=64)


class Net(nn.Module):  # 继承pytorch的nn.Module类
    def __init__(self):  # 初始化参数及功能
        super(Net, self).__init__()  # 继承时必写的
        self.flatten = Flatten()  # 定义的功能，其中Flatten()是从torch.nn导入的

    def forward(self, x):  # 实现功能
        y = self.flatten(x)
        return y

net = Net()  # 实例化

for data in train_loader:  # 取数据
    imgs, targets = data
    print("原：", imgs.shape)
    result = net(imgs)  # 计算
    print("flatten后：", result.shape)
