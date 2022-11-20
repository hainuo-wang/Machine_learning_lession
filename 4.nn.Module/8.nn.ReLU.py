import torch
from torch import nn
from torch.nn import ReLU

ipt = torch.tensor([[1, -0.5],
                    [-1, 3]])
ipt = torch.reshape(ipt, (-1, 1, 2, 2))
print(ipt.shape)


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.relu1 = ReLU(inplace=False)

    def forward(self, x):
        y = self.relu1(x)
        return y


net = Net()
opt = net(ipt)
print(opt)
