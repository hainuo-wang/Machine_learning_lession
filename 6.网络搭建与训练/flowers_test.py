import json

import torch
import torchvision
from PIL import Image
from torch import nn

# 定义训练设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 网络、损失函数，数据
# 需要预测的图片地址
image_path = "../data/flowers/roses/110472418_87b6a3aa98_m.jpg"
# 打开图片
image = Image.open(image_path)
# 和训练过程一样的transform
transform = torchvision.transforms.Compose([
    torchvision.transforms.Resize(256),
    torchvision.transforms.CenterCrop(225),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
])
image = transform(image)
# 对图片进行维度变换，具体的可以去搜资料
image = torch.unsqueeze(image, dim=0)
# 之前读取数据时的json文件，保存着类别的编码
json_path = 'class_indices.json'
with open(json_path, "r") as f:
    class_indict = json.load(f)


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.pool3 = nn.MaxPool2d(kernel_size=2)
        self.relu = nn.ReLU()
        self.flt = nn.Flatten()
        self.fc1 = nn.Linear(in_features=100352, out_features=625)
        self.fc2 = nn.Linear(in_features=625, out_features=5)

    def forward(self, x):
        x = self.pool1(self.relu(self.conv1(x)))
        x = self.pool2(self.relu(self.conv2(x)))
        x = self.pool3(self.relu(self.conv3(x)))
        x = self.flt(x)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# 实例化网络
net = Net().to(device)
# 训练参数导入
net.load_state_dict(torch.load('./weights/model-9.pth', map_location=device))
# 预测
net.eval()
with torch.no_grad():
    output = torch.squeeze(net(image.to(device))).cpu()
    predict = torch.softmax(output, dim=0)
    predict_cla = torch.argmax(predict).numpy()
print_res = "class: {}   prob: {:.3}".format(class_indict[str(predict_cla)],
                                             predict[predict_cla].numpy())
for i in range(len(predict)):
    print("class: {:10}   prob: {:.3}".format(class_indict[str(i)],
                                              predict[i].numpy()))
