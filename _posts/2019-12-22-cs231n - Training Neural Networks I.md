---
layout: post
title: cs231n - Training Neural Networks I
subtitle: Training Neural Networks, part I [slides]
date: 2019-12-22
author: kevin
header-img: img/green-bg.jpg
catalog: true
tags:
    - deep learning
    - reading notes
---



## preface



之前介绍了那么多的理论知识，现在开始终于要着手训练神经网络了，这里就记录一下相关的知识点



## 训练流程



训练一个神经网络通常有以下几个流程：1. 训练前设置好相关的参数，如激活函数，预处理，权重初始化，正则化，检查梯度；2. 开始训练模型，再训练的过程中观察学习的过程，以及参数的更新和超参数优化问题；3. 评估模型，测试模型的准确率和帧率。下面我们就一个一个的来说说



## Activation Functions



在[之前的文章](https://szukevin.site/2019/11/09/cs231n-Neural-Networks/)中我们就已经初步了解了激活函数的作用了，也就是一个非线性的函数，如果没有激活函数的话，神经网络层数再高也只是个线性的矩阵相乘而已，用了激活函数就使得神经网络的非线性成分增多，能让网络学习更加复杂的东西，这里详细价绍几个激活函数的特性以及方程



### sigmoid



$\alpha \in A$


$$
\begin{align*}x &= 1, & y &= 2, && \text{initialize}\\z &= 3, & w &= 4,\\
\text{some more text, and}a &= 5, & b &= 5.\end{align*}
$$





