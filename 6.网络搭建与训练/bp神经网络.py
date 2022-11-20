import numpy as np
import torch

xy = np.loadtxt('../data/bp神经网络.txt', delimiter='	', dtype=np.float32)
x_data = torch.from_numpy(xy[:, :-1])
y_data = torch.from_numpy(xy[:, [-1]])
mean, std = torch.mean(x_data), torch.std(x_data)
x_data = (x_data-mean)/std
mean, std = torch.mean(y_data), torch.std(y_data)
y_data = (y_data-mean)/std
x_data = torch.nn.functional.normalize(x_data, dim=0)
y_data = torch.nn.functional.normalize(y_data, dim=0)


class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.linear1 = torch.nn.Linear(7, 4)
        self.linear2 = torch.nn.Linear(4, 2)
        self.linear3 = torch.nn.Linear(2, 1)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        x = self.sigmoid(self.linear1(x))
        x = self.sigmoid(self.linear2(x))
        x = self.sigmoid(self.linear3(x))
        return x


model = Model()
criterion = torch.nn.BCELoss(size_average=True)  # 损失函数
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)  # 优化函数，随机梯度递减

for epoch in range(100):
    # 前馈
    y_pred = model(x_data)
    loss = criterion(y_pred, y_data)
    print(epoch, loss.item())

    # 反馈
    optimizer.zero_grad()
    loss.backward()

    # 更新
    optimizer.step()
