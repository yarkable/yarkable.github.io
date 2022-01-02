---
layout: post
title: matplotlib个人手册
subtitle: 
date: 2020-03-26
author: kevin
header-img: img/green-bg.jpg
catalog: true
tags:
    - deep learning
    - python
    - matplotlib
---



## 基本使用



> tips: 在 jupyter notebook 中加上一行 %matplotlib inline 可以使图像显示出来



### 画一张图



`plt.plot()` 方法可以将给定的数据绘制成图片，再用 `plt.show()` 将图片展示出来

```python
import matplotlib.pyplot as plt
import numpy as np
x = np.linspace(0, 2 * np.pi, 100)
y = np.sin(x)
plt.plot(x, y)
plt.show()
```



### 通过 figure 定义每张图像



和 matlab 是一样的，matplotlib 也用 `plt.figure()` 来指定一张图片，声明一张图片后，后续的操作都是针对该张图片，figure() 可指定参数 `num` ，默认是从1 开始的顺序数

```python
import matplotlib.pyplot as plt
import numpy as np
x = np.linspace(0, 2 * np.pi, 100)
y1 = np.sin(x)
y2 = np.cos(x)
plt.figure()
plt.plot(x, y1)
plt.figure(6)
plt.plot(x, y2)
plt.show()
```



### 一张图显示多个函数



挺简单的，其实就是上面说的用一个 figure，然后所有的 plot 操作都在这张 figure 上进行，所以就相当于在一张图上显示了很多个函数

```python
import matplotlib.pyplot as plt
import numpy as np
x = np.linspace(0, 2 * np.pi, 100)
y1 = np.sin(x)
y2 = np.cos(x)
plt.figure()
plt.plot(x, y1)
plt.plot(x, y2)
plt.show()
```





## 设置坐标轴







## 多图合并



一般在代码中想让多张图在一张图上面显示的话呢，可以用下面这种代码，也就是 `fig,axes = plt.subplots()` ，这样得到了两个列表，只需要对 axes 列表进行操作就能够得到相应的图像



另外，`plt.imshow()` 函数里面的值要么是 0-255 的整数，要么就是 0-1 之间的浮点数，否则会显示失败，报错

```python
fig, axes = plt.subplots(nrows=1, ncols=3, figsize=[12.8, 7.2], dpi=300)
axes[0].axis('off')
import os.path as osp
img = img / np.amax(img)
img = np.clip(img, 0, 1)        
axes[0].imshow(img)
axes[0].set_title("Original Img")
for i in range(gt_bboxes.shape[0]):
    bbox = gt_bboxes[i, :4].cpu().numpy()
    rect = plt.Rectangle((bbox[0], bbox[1]), bbox[2] - bbox[0], bbox[3] - bbox[1], linewidth=1, edgecolor='g', facecolor='none')
    axes[0].add_patch(rect)

    axes[1].axis('off')
    for i in range(gt_bboxes.shape[0]):
        bbox = gt_bboxes[i, :4].cpu().numpy()
        rect = plt.Rectangle((bbox[0], bbox[1]), bbox[2] - bbox[0], bbox[3] - bbox[1], linewidth=1, edgecolor='g', facecolor='none')
        axes[2].add_patch(rect)
    for i in range(pos_points.shape[0]):
        axes[2].scatter(pos_points[i, 0], pos_points[i, 1], s=3,linewidths=1)
        axes[2].set_title("Positive Points")
        plt.savefig('./show_dirs/bool_mask_with_points{}.png'.format(time.time()), bbox_inches = 'tight')
```



## matplotlib 画图出现重叠



在 for 循环中调用这个函数时会导致第二次循环时绘制的图是在第一次绘图的基础上绘制的，这就出现了后面保存的图中数据越来越多。该问题主要是 matplotlib 会记录之前的画图，在每次使用完后，应该调用 `plt.clf()` 函数。



## reference





https://morvanzhou.github.io/tutorials/data-manipulation/plt/

[使用matplotlib绘图时出现数据重复重叠问题_twinkle-zp的博客-CSDN博客](https://blog.csdn.net/muchen123456/article/details/106041525)