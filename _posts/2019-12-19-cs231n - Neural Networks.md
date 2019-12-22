---
layout: post
title: cs231n - Neural Networks
subtitle: Introduction to Neural Networks [slides]
date: 2019-12-19
author: kevin
header-img: img/green-bg.jpg
catalog: true
tags:
    - deep learning
    - reading notes
---



## preface



之前介绍了 KNN 和线性分类器，这次终于上到神经网络了，这一节举了很多例子来说明神经网络的数学推导和反向传播的公式，如果忘记的话可以再回去看看



## Neural Networks



在这之前，我们的分类器用的都是一个线性函数 `f = Wx + b` ，然而在神经网络这里，表达式也是差不多的，他稍微变化了一个形式，`f = W2max(0, W1x)` ，这里是一个两层的神经网络，其中的 `max(0, W1x)` 就是第一层神经网络的输出，因此多层神经网络的公式也就是不断地进行上面的操作

![nn.jpg](https://i.loli.net/2019/12/22/BDl2onVMLh4ONqb.jpg)

设想一下，要是我们不用后面的 `max(0, W1x)` 而是用 `W1x` 的话，公式就变成了 `f = W2W1x` ，这又是一个线性的分类器了，所以不能这样子，神经网络是个非线性的分类器。而 `max(0, W1x)` 这玩意被叫做激活函数（activation function），就是选择哪些神经元可以被激活，造成一种非线性的效果，激活函数的种类有很多种，其中 `ReLU(Rectified Linear Unit)` 是被用的最多的，如果不知道该用哪个激活函数的话就用 ReLU 吧



![relu.jpg](https://i.loli.net/2019/12/23/nRti8DZLGkOQaWy.jpg)



然后下面是一个三层神经网络的结构和代码表述（基础知识就不讲太细了）

![three-layer-nn.jpg](https://i.loli.net/2019/12/23/291EWgOzp5Ni6DJ.jpg)



然后这是一个单独的神经元的长相，他长这样，就用输入的样本和每一个神经元的点积之和加上偏置，最后再通过一个激活函数进行输出

![neuron.jpg](https://i.loli.net/2019/12/23/EfnS4lDTuv6a2VO.jpg)



## How to compute gradients



下面说说怎么样去计算神经网络的梯度，还是跟线性分类器那里一样按部就班来整一套，先给出一个 score function ，再给出一个 loss function ，其中 loss function 又包括了预测错误的损失和正则化的损失，不过再线性分类器中只有一个 W 的正则化损失，神经网络中每个神经元都包含了一个 W ，因此在计算正则化损失的时候得累加起来，然后用这个损失函数对 W 进行求偏导，然后就可以学习 W 这个参数



计算梯度的话，首先想到的可能就是手动推导，不过这样子非常 silly ，需要计算大量的矩阵运算，并且如果将损失函数由 SVM loss function 换成 softmax 的话又得重新运算一遍，最致命的问题，现在的网络模型可以变得很大，因此用这种方法计算梯度几乎是不行的。所以，另外一种更好的方法就是画出神经网络的计算流图，然后通过反向传播求得梯度

![backwards-propagation.jpg](https://i.loli.net/2019/12/23/krnJg52xWNyA63G.jpg)



### Backpropagation



虽然已经知道反向传播是个啥东西了，但还是用一个简单的例子来回顾一下，随便设几个常数

![sample.jpg](https://i.loli.net/2019/12/23/IixZ6tsKV8A2WBq.jpg)

通过链式法则就可以求得最终的 f 对各个参数节点的偏导数

![chain-rule.jpg](https://i.loli.net/2019/12/23/N6gHM8PWmJCZLI9.jpg)



这里有给出一个复杂点的反向传播的例子，就是不断地用梯度相乘，不明白的可以再去看看对应的 PPT ，里卖弄讲解得非常详细

![graph.jpg](https://i.loli.net/2019/12/23/AspkdU1bKcLXeiO.jpg)



并且给出了常见的计算流图的反向传播的值的计算方法如下图（add, mul, copy, max）



![gradient-flow.jpg](https://i.loli.net/2019/12/23/eKlFxwRcBAGdvOt.jpg)



并且还将正向传播和反向传播的代码给写出来配合计算流图讲解，印象更加深刻，总之就是求偏导加上链式法则那一套咯

![forward-back.jpg](https://i.loli.net/2019/12/23/ARHDqEUyuXZMVY2.jpg)



### Backprop with vector-valued functions



