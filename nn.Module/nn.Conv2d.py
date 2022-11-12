# nn.Conv2d在nn.Module类里面
# nn.Conv2d不需要我们自己去定义卷积核的
# 我们只需要告诉它输入通道数，输出通道数，卷积核大小……
# 就可以自行来进行卷积运算
import torch
from torch import nn

x = torch.rand(1, 1, 28, 28)  # 输入图像
layer = nn.Conv2d(1, 3, kernel_size=3)  # 实例化nn.Conv2d
result = layer.forward(x)  # 调用nn.Module的实例方法：forward
print(result.shape)
