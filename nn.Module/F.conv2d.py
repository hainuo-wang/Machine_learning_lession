# F.conv2d不属于nn.Module
# 让大家了解卷积运算的一个过程
import torch
import torch.nn.functional as F

ipt = torch.tensor([
    [1, 2, 0, 3, 1],
    [0, 1, 2, 3, 1],
    [1, 2, 1, 0, 0],
    [5, 2, 3, 1, 1],
    [2, 1, 0, 1, 1]
])
kernel = torch.tensor([
    [1, 2, 1],
    [0, 1, 0],
    [2, 1, 0]
])
ipt = torch.reshape(ipt, (1, 1, 5, 5))
kernel = torch.reshape(kernel, (1, 1, 3, 3))
result = F.conv2d(ipt, kernel)
print(result)