import os
import shutil

from PIL import Image
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

logs_path = "logs"
if os.path.exists(logs_path):
    shutil.rmtree(logs_path)
writer = SummaryWriter(logs_path)

path = '../dataset_and_dataloader/hymenoptera_data/bees/16838648_415acd9e3f.jpg'
img = Image.open(path)
print(img)
# img.show()
# ToTensor 此操作必须放在基本变换（放缩和裁剪等）后
# PIL -> tensor并且归一化
trans_ToTensor = transforms.ToTensor()
img_tensor = trans_ToTensor(img)
# writer.add_image('ToTensor', img_tensor)
# writer.close()
# Normalize 标准化 此操作必须在ToTensor之后
# print(img_tensor[0][0][0])
trans_norm = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                  std=[0.229, 0.224, 0.225])
img_norm = trans_norm(img_tensor)
# writer.add_image('Normalize', img_norm)
# writer.close()
transform = transforms.Compose(
            [transforms.Resize(256),  # 放缩
             transforms.CenterCrop(224),  # 中心裁剪
             transforms.ToTensor(),  # 类型转换并且归一化
             transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                  std=[0.229, 0.224, 0.225])])  # 标准化
img_final = transform(img)
writer.add_image('Compose', img_final)
writer.close()
