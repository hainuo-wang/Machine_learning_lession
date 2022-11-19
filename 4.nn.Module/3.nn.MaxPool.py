import torch
from torch import nn
from torch.nn import MaxPool2d

ipt = torch.tensor([
    [1, 2, 0, 3, 1],
    [0, 1, 2, 3, 1],
    [1, 2, 1, 0, 0],
    [5, 2, 3, 1, 1],
    [2, 1, 0, 1, 1]
], dtype=torch.float32)
ipt = torch.reshape(ipt, (-1, 1, 5, 5))


class Net(nn.Module):  # 继承pytorch的nn.Module类
    def __init__(self):  # 初始化参数及功能
        super(Net, self).__init__()  # 继承时必写的
        self.maxpool = MaxPool2d(kernel_size=3, stride=1)  # 定义的功能，其中MaxPool2d()是从torch.nn导入的
        # MaxPool2d默认步长是2

    def forward(self, x):  # 实现功能
        y = self.maxpool(x)
        return y


net = Net()  # 实例化
result = net(ipt)  # 计算
print(result)
