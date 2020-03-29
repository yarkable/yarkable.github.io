---
layout: post
title: pytorch训练kaggle猫狗大战分类器
subtitle: 
date: 2020-03-18
author: kevin
header-img: img/green-bg.jpg
catalog: true
tags:
    - deep learning
    - python
    - pytorch
---



## preface



这篇文章来写一下用 pytorch 训练的一个 CNN 分类器，数据集选用的是 kaggle 上的猫狗大战数据集，只有两个 class ，不过数据集还是挺多的，足够完成我们的分类任务。这份数据集分为 `train` 和 `test` 两个文件夹，装着训练集和测试集，还有一个 `sample_submission.csv` 用来提交我们训练的模型在测试集上的分类情况。值得注意的是，训练集是带标签的，标签在文件名中，如 `cat.7741.jpg`，而测试集是不带标签的，因为我们模型在测试集中测试后分类的结果是要填到 csv 文件中提交的，所以不能拿测试集来评估模型，我们可以在训练集中划分出一个验证集来评估模型。



## 划分数据集



首先这是我们需要的所有的模块，缺少的可以用 pip 安装

```python
from torchvision.models.resnet import resnet18
import os
import random
from PIL import Image
import torch.utils.data as data
import numpy as np
import torchvision.transforms as transforms
import torch
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import *
import torchvision.transforms as transforms
import numpy as np
import os
```



深度学习的背后是依靠着大量标注好的数据集，所以我们要首先将数据集给整好，在 pytorch 中通过继承 `torch.utils.data.Dataset` 类来实现一个数据集的类，这里我们的训练集和验证集的比例是 7：3

```python
class DogCat(data.Dataset):
    def __init__(self, root, transform=None, train=True, val=False):
        self.val = val
        self.train = train
        self.transform = transform
        # imgs为一个储存了所有数据集绝对路径的列表
        imgs = [os.path.join(root, img) for img in os.listdir(root)]    
 
        if self.val:
            imgs = sorted(imgs, key=lambda x: int(x.split('.')[-2].split('/')[-1]))
        else:
            # 根据图片的num排序，如 cat.11.jpg -> 11
            imgs = sorted(imgs, key=lambda x: int(x.split('.')[-2]))
        imgs_num = len(imgs)
        random.shuffle(imgs) # 打乱顺序
        if self.train:
            self.imgs = imgs[:int(0.7 * imgs_num)]
        else:
            self.imgs = imgs[int(0.7 * imgs_num):]
 
    # 作为迭代器必须有的方法
    def __getitem__(self, index):
        img_path = self.imgs[index]
        label = 1 if 'dog' in img_path.split('/')[-1] else 0  # 狗的label设为1，猫的设为0
        data = Image.open(img_path)
        data = self.transform(data)
        return data, label
 
    def __len__(self):
        return len(self.imgs)
```



在 `torchvision.transforms.Compose` 类里有很多针对图像的变化，包括图像旋转以及随即裁剪等增强方式，注意我们这里输入的图片是三维的 PIL image，所以要用 `ToTensor()` 方法将其转化成 pytorch 的 tensor 形式

```python
# 对数据集训练集的处理，其实可以直接放到 DogCat 类里面去
transform_train = transforms.Compose([
    transforms.Resize((256, 256)),  # 先调整图片大小至256x256
    transforms.RandomCrop((224, 224)),  # 再随机裁剪到224x224
    transforms.RandomHorizontalFlip(),  # 随机的图像水平翻转，通俗讲就是图像的左右对调
    transforms.ToTensor(), # Convert a ``PIL Image`` or ``numpy.ndarray`` to tensor.
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.2225))  # 归一化，数值是用ImageNet给出的数值
])
 
# 对数据集验证集的处理
transform_val = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
])
```



最后用上面我们定义好的数据集类生成训练集对象和验证集对象，这里我们给了 `batch_size` 为 20，每 20 个数据进行一次梯度下降，其实一般 `batch_size` 用 2 的整数次方比较好，`num_works` 是加载数据用几个线程的意思，在 windows 上要将这个参数去掉，否则会报错

```python
# 生成训练集和验证集
trainset = DogCat('/data/rpcv/kevin/dataset/dogs-vs-cats-redux-kernels-edition/train', transform=transform_train)
valset = DogCat('/data/rpcv/kevin/dataset/dogs-vs-cats-redux-kernels-edition/train', transform=transform_val, train=False, val=True)
# 将训练集和验证集放到 DataLoader 中去，shuffle 进行打乱顺序（在多个 epoch 的情况下）
# num_workers 加载数据用多少的子线程（windows不能用这个参数）
trainloader = torch.utils.data.DataLoader(trainset, batch_size=20, shuffle=True, num_workers=1)
valloader = torch.utils.data.DataLoader(valset, batch_size=20, shuffle=False, num_workers=1)
```



## 定义网络



我们直接用 ResNet18 来作为基础结构了，在他的基础上进行迁移学习，因为 ResNet 曾经在 ImageNet 上训练过，并且拿到了第一名的成绩，所以利用它的权重来训练的效果会比较好。定义网络是通过继承 `torch.nn.Module` 来实现的，这里由于我们是更改 ResNet ，所以在 `__init__()` 初始化时要传入一个 model 参数。



接着将 ResNet 最后面的一层全连接层去掉改成我们的全连接层，因为 ResNet 在 ImageNet 训练时有 1000 个类，所以它最后全连接输出的是 1000，而我们只要输出 2 就行了。至于 512 是怎么来的呢，全连接上一层的输出是 512，这个可以通过 `print(Net)` 来看网络的每一层结构，后面会说。



因为我们重写了 `forward()` 函数，所以要加上 `super().__init__()` ，而用 `x.view(x.size(), -1)` 是因为 x 后面接全连接层，这一层的输出是一个 feather-map，要将其所有的特征都变成一个一维向量才能够被全连接层接受，`x.size(0)` 指的是 `batch_size` 



![x.view](https://i.loli.net/2020/03/30/Vp8m9uM4gESOGNx.png)



```python
class Net(nn.Module):
    def __init__(self, model):
        super(Net, self).__init__()
        # 去掉model的最后1层
        self.resnet_layer = nn.Sequential(*list(model.children())[:-1])
        self.Linear_layer = nn.Linear(512, 2) #加上一层参数修改好的全连接层
 
    def forward(self, x):
        x = self.resnet_layer(x)
        x = x.view(x.size(0), -1)
        x = self.Linear_layer(x)
        return x
```



## 定义训练测试函数



训练函数如下，训练前要将模型调成训练模式 `model.train()`，然后就初始化 loss 和 精确度，接着将数据集加载进来，这里一次加载是 batch_size 个数据以及标签，所以 image 的个数是 20，

```python
def train(epoch):
    print('\nEpoch: %d' % epoch)
#     scheduler.step()
    model.train()
    train_acc = 0.0
    for batch_idx, (img, label) in enumerate(trainloader): # 迭代器，一次迭代 batch_size 个数据进去
        image = img.to(device)
        label = label.to(device)
        optimizer.zero_grad()
        out = model(image)
        loss = criterion(out, label)
        loss.backward()
        optimizer.step()
        train_acc = get_acc(out, label)
        print("Epoch:%d [%d|%d] loss:%f acc:%f" % (epoch, batch_idx, len(trainloader), loss.mean(), train_acc))
```

