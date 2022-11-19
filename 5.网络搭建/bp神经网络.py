from torch import nn
from torch.nn import ReLU, Sigmoid, Linear

# 8列数据的列表
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.linear1 = Linear(in_features=8, out_features=32)
        self.linear2 = Linear(in_features=32, out_features=8)
        self.sigmoid = Sigmoid()

    def forward(self, x):
        x = self.linear1(x)
        x = self.sigmoid(x)
        x = self.linear2(x)
        x = self.sigmoid(x)
        return x
