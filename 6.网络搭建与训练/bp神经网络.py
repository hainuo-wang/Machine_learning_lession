import numpy as np
import torch
from sklearn import preprocessing
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split

# 数据读取
xy = np.loadtxt('../data/Real estate valuation data set.txt', delimiter='	', dtype=np.float32)
x_data = torch.from_numpy(xy[:, :-1])
y_data = torch.from_numpy(xy[:, [-1]])
# 归一化
mean, std = torch.mean(x_data), torch.std(x_data)
x_data = (x_data - mean) / std
mean, std = torch.mean(y_data), torch.std(y_data)
y_data = (y_data - mean) / std
# 标准化
x_data = torch.nn.functional.normalize(x_data, dim=0)
y_data = torch.nn.functional.normalize(y_data, dim=0)
# 数据集分割
x_train, x_test = train_test_split(x_data, test_size=0.2)
y_train, y_test = train_test_split(y_data, test_size=0.2)


# 网络定义
class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.linear1 = torch.nn.Linear(6, 4)
        self.linear2 = torch.nn.Linear(4, 2)
        self.linear3 = torch.nn.Linear(2, 1)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        x = self.sigmoid(self.linear1(x))
        x = self.sigmoid(self.linear2(x))
        x = self.sigmoid(self.linear3(x))
        return x


# 实例化
model = Model()
criterion = torch.nn.BCELoss(size_average=True)  # 损失函数
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)  # 优化函数，随机梯度递减

for epoch in range(100):
    # 前馈
    y_pred = model(x_train)
    loss = criterion(y_pred, y_train)
    print("--------epoch{}---------".format(epoch))
    print("loss", loss.item())
    # 反馈
    optimizer.zero_grad()
    loss.backward()
    # 更新
    optimizer.step()
    correct = 0
    total = 0
    with torch.no_grad():  # 下面是没有梯度的计算,主要是测试集使用，不需要再计算梯度了
        outputs = model(x_test)
        print("均方误差", mean_squared_error(y_test, outputs))
        print("平方绝对误差", mean_absolute_error(y_test, outputs))
        print("------------------------")
