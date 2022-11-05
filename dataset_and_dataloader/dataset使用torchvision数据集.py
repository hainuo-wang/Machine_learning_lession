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

