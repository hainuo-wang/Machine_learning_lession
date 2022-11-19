import torch
import torchvision
from torch import nn
from torch.nn import Linear
from torch.utils.data import DataLoader

# 加载
train_dataset = torchvision.datasets.CIFAR10('../2.dataset_and_dataloader/CIFAR10', train=True,
                                             transform=torchvision.transforms.ToTensor(), download=True)
test_dataset = torchvision.datasets.CIFAR10('../2.dataset_and_dataloader/CIFAR10', train=False,
                                            transform=torchvision.transforms.ToTensor(), download=True)
# 打包
train_loader = DataLoader(train_dataset, batch_size=64)
val_loader = DataLoader(test_dataset, batch_size=64)


class Net(nn.Module):  # 继承pytorch的nn.Module类
    def __init__(self):  # 初始化参数及功能
        super(Net, self).__init__()  # 继承时必写的
        self.linear = Linear(196608, 10)  # 定义的功能，其中Linear()是从torch.nn导入的

    def forward(self, x):  # 实现功能
        y = self.linear(x)
        return y

net = Net()  # 实例化

for data in train_loader:  # 取数据
    imgs, targets = data
    print("原：", imgs.shape)
    opt = torch.flatten(imgs)
    print("flatten后：", opt.shape)
    result = net(opt)  # 计算
    print("linear后：", result.shape)
