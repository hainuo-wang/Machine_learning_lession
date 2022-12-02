import torchvision.transforms
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from utils.read_data import read_split_data
from utils.MyDataset import MyDataSet

train_images_path, train_images_label, val_images_path, val_images_label = read_split_data('../data/hymenoptera_data')
# print(train_images_label[0])
transform = torchvision.transforms.ToTensor()
train_dataset = MyDataSet(train_images_path, train_images_label, transform=transform)  # 加载数据集（单个的图片以及标签）
val_dataset = MyDataSet(val_images_path, val_images_label, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=64)
val_loader = DataLoader(val_dataset, batch_size=64)

writer = SummaryWriter('log1')
for i in range(10):
    img, label = train_dataset[i]
    writer.add_image('set2', img, i)
writer.close()

# tensorboard使用方法：复制writer = SummaryWriter('log1')中log1的绝对地址，然后
# 在终端（terminal）输入：tensorboard --logdir=绝对地址
# 比如我这里就是：tensorboard --logdir=D:\project\pycharm\Machine_learning_lession\2.dataset_and_dataloader\log1
# 然后回车，点击有6006的网址即可进入tensorboard界面
