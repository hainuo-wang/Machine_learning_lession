from torch.utils.data import DataLoader
from read_data import read_split_data
from MyDataset import MyDataSet

train_images_path, train_images_label, val_images_path, val_images_label = read_split_data('hymenoptera_data')
print(train_images_label[0])
train_dataset = MyDataSet(train_images_path, train_images_label)
val_dataset = MyDataSet(val_images_path, val_images_label)
train_loader = DataLoader(train_dataset, batch_size=64)
val_loader = DataLoader(val_dataset, batch_size=64)
