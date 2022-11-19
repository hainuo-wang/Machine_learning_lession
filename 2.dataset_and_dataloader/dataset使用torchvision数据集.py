import torchvision
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader

train_dataset = torchvision.datasets.CIFAR10('CIFAR10', train=True,
                                             transform=torchvision.transforms.ToTensor(), download=True)
test_dataset = torchvision.datasets.CIFAR10('CIFAR10', train=False,
                                            transform=torchvision.transforms.ToTensor(), download=True)
print(test_dataset[0])
img, label = test_dataset[0]
print(test_dataset.classes)
print(img)
print(label)
writer = SummaryWriter('log1')
for i in range(10):
    img, label = test_dataset[i]
    writer.add_image('set1', img, i)
writer.close()
train_loader = DataLoader(train_dataset, batch_size=64)
test_loader = DataLoader(test_dataset, batch_size=64)

# tensorboard使用方法：复制writer = SummaryWriter('log1')中log1的绝对地址，然后
# 在终端（terminal）输入：tensorboard --logdir=绝对地址
# 比如我这里就是：tensorboard --logdir=D:\project\pycharm\Machine_learning_lession\2.dataset_and_dataloader\log1
# 然后回车，点击有6006的网址即可进入tensorboard界面

