import torchvision
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from utils.read_data import read_split_data
from utils.MyDataset import MyDataSet

train_images_path, train_images_label, val_images_path, val_images_label = read_split_data('hymenoptera_data')
print(train_images_label[0])
transform = torchvision.transforms.Compose([torchvision.transforms.Resize((224, 224)),
                                            torchvision.transforms.ToTensor()])
train_dataset = MyDataSet(train_images_path, train_images_label, transform=transform)
val_dataset = MyDataSet(val_images_path, val_images_label, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=64)
val_loader = DataLoader(val_dataset, batch_size=64)

writer = SummaryWriter('log2')
step = 0
for data in val_loader:
    imgs, targets = data
    writer.add_images('val_data', imgs, step)
    step += 1
writer.close()
