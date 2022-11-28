import torch
from torch import nn

ipt = torch.tensor([1, 2, 3], dtype=torch.float32)
targets = torch.tensor([1, 3, 3], dtype=torch.float32)

ipt = torch.reshape(ipt, (1, 1, 1, 3))
targets = torch.reshape(targets, (1, 1, 1, 3))

loss = nn.L1Loss()
result = loss(ipt, targets)
print(result)
