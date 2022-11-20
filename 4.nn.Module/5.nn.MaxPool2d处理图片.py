import torchvision
from torch import nn
from torch.nn import MaxPool2d
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
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
        self.maxpool = MaxPool2d(kernel_size=3, stride=1)  # 定义的功能，其中MaxPool2d()是从torch.nn导入的
        # MaxPool2d默认步长是2

    def forward(self, x):  # 实现功能
        y = self.maxpool(x)
        return y

net = Net()  # 实例化

writer = SummaryWriter("logs1")  # 定义tensorboard
step = 0  # 提供横坐标
for data in train_loader:  # 取数据
    imgs, targets = data
    opt = net(imgs)  # 计算
    # opt = torch.reshape(opt, (-1, 3, 30, 30))  # 因为tensorboard的要求
    writer.add_images("conv", opt, step)
    step += 1
writer.close()
