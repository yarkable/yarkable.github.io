---
layout: post
title: cs231n - Deep Learning Hardware and Software
subtitle: CPUs, GPUs, TPUs, PyTorch, TensorFlow
date: 2019-12-21
author: kevin
header-img: img/green-bg.jpg
catalog: true
tags:
    - deep learning
    - reading notes
---



## preface



这一节就来讲讲深度学习的一些软件和硬件相关的内容，包括 CPU 和 GPU ，以及一些主流的深度学习框架



## CPU & GPU



这里就是一个机箱里面 GPU 和 CPU 的安装位置，连我这个装机小白都清楚这些，看见绿色的 GEFORCE 就是财富的象征

![device-cpu.jpg](https://i.loli.net/2019/12/23/yviVtY1aRKXsfjS.jpg)

训练神经网络就是吃显卡的性能，所以一台好的工作站通常会有好几块高性能的显卡

![device-gpu.jpg](https://i.loli.net/2019/12/23/Fy2SgsXCzTmcGYq.jpg)

我们的训练数据放在硬盘中，模型在显卡中，因此要从硬盘中读取数据去训练，如果处理得不好的话，这部分可能会成为训练的瓶颈，所以我们一般用 SSD 而不是 HDD ，并且将所有的数据读到 RAM 中，并且用多线程 CPU 来预先取得数据

![communication.jpg](https://i.loli.net/2019/12/23/tPSOdiGfue1H6QZ.jpg)



## numpy & pytorch



随着技术的发展，已经有很多神经网络的框架出现，像 caffe ，tensorflow ，pytorch ，darknet 等等都是比较主流的框架，numpy 是 python 中用来做向量计算的库，底层用 C++ 实现，会提高运算的速度。用 numpy 实现一个神经网络的结构也是可以的，写起来很轻松，但是会有一些不方便的地方，比如用不了 GPU 加速，并且梯度也很难计算

![numpy.jpg](https://i.loli.net/2019/12/23/iNT38UcBRMwhuja.jpg)



用 pytorch 的话，语法和 numpy 非常相似，并且可以自动求梯度，而且可以指定使用 GPU 加速，因此我最喜欢的一个框架就是 pytorch 了，非常容易上手

![pytorch.jpg](https://i.loli.net/2019/12/23/SKE6DGoVkCzg5M3.jpg)



有关 pytorch 更加具体的内容会在后面实际训练的时候再介绍，网上也有很多的 demo 可以参照，框架那么多，选择一款最适合自己的就好了，有些人也更加喜欢 tensorflow，都没问题，开心就好。