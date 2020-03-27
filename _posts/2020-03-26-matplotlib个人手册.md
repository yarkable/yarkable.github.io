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



