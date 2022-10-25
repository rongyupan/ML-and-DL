from json import load
import torch
import torchvision
from torch import nn
from torchvision import transforms, models, datasets
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

device = torch.device("cuda:0")

def load_model(pth_name):
  model = models.resnet152(pretrained=True)
  for param in model.parameters():
    param.requires_grad = False
  num_ftrs = model.fc.in_features
  model.fc = nn.Sequential(nn.Linear(num_ftrs, 102), nn.LogSoftmax(dim=1))
  input_size = 224

  model = model.to(device)

  checkpoint = torch.load(pth_name)
  best_acc = checkpoint["best_acc"]
  model.load_state_dict(checkpoint["state_dict"])

  return model

def process_image(image_path):
  img = Image.open(image_path)
  # thumbnail只能缩小，仍会保持原图的宽高比
  # 如果输入的宽高比和原图不同，则按照最小对应边进行比例缩小
  if img.size[0] > img.size[1]:
    img.thumbnail((10000,256))
  else:
    img.thumbnail((256,10000))
  left_margin = (img.width-224)/2
  bottom_margin = (img.height-224)/2
  right_margin = left_margin + 224
  top_margin = bottom_margin + 224
  img = img.crop((left_margin, bottom_margin, right_margin, top_margin))

  img = np.array(img)/255
  mean = np.array([0.485, 0.456, 0.406])
  std = np.array([0.229, 0.224, 0.225])
  img = (img-mean)/std

  # 颜色通道应该放在首位
  img = img.transpose([2,0,1])
  
  return img

def imshow(image, ax=None, title=None):
  if ax is None:
    fig, ax = plt.subplots()
  
  # 颜色通道还原
  image = np.array(image).transpose((1,2,0))
  
  # 预处理还原
  mean = np.array([0.485, 0.456, 0.406])
  std = np.array([0.229, 0.224, 0.225])
  image = std*image + mean

  # 把数值范围限定到 0-1 之间
  image = np.clip(image, 0, 1)

  ax.imshow(image)
  ax.set_title(title)
  plt.savefig("image.jpg")

  return ax


if __name__ == '__main__':

  model = load_model("./wz.pth")

  image_path = "./flower_data/train/84/image_02580.jpg"
  img = process_image(image_path)
  imshow(img)

  img = torch.from_numpy(img)
  # 转换成 float 格式；
  # 模型的输入参数是 Batch-Chanel-Height-Width 
  # unsqueeze 插入一个维度
  img = img.float().unsqueeze(0)
  output = model(img.cuda())
  print(output.shape)

  _, preds_tensor = torch.max(output, 1)
  preds = np.squeeze(preds_tensor.cpu().numpy())

  print(preds)
  