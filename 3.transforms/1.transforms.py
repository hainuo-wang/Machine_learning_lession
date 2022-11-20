from PIL import Image
from torchvision import transforms

path = '../data/hymenoptera_data/bees/16838648_415acd9e3f.jpg'
with open(path, 'rb') as f:
    img = Image.open(f)
    img_parse = img.convert('RGB')
# img_parse.show()
# transform = 3.transforms.Resize((224, 128))  # 放缩
transform = transforms.CenterCrop((128, 128))  # 裁剪
img_new = transform(img_parse)
img_new.show()
