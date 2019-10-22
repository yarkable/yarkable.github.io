---

layout:     post
title:      cs231n - linear classifiers
subtitle:   
date:       2019-10-19
author:     kevin
header-img: img/green-bg.jpg
catalog: true
tags:
    - deep learning
	- reading notes
---






上节课讲的 `KNN` 效率很低，并不能满足大多数情况，主要体现在

1. 分类器要记住所有样本的特征，但是今天的样本量可以达到几个 GB ，而且维度也会是以千万计。
2. 要将测试集与所有的训练集进行比较，效率十分慢。

因此，用线性分类器这种方法来满足大多数情况下的分类，因为他只需要训练 W 和 b 参数，一旦训练好了，就可以把训练集给扔了，并且也不用与每一个训练集进行比较，只要做一些简单的矩阵相乘相加就行。



---

线性分类器通过设计 `Score function` 和 `Loss function` 来实现分类， 函数原型如下



```
f(xi, W, b) = Wxi + b
```



* xi 是输入样本的所有特征，假设输入是一个 `4*4*3` 的样本，xi 就是他们展开的一个 `48*1` 的列向量

* W 是权重（Weight），是一个 `C*D` 的向量，C 是分类器最终输出的类型的个数，D 是一个样本中的特征数量，其实 W 就相当于是用 C 张图片去和输入的图片进行内积操作，选出得分（Score）最大的一个类别，最终分类的结果就是这个类。W 的每一排都相当于是一张模板图片展开，如下图给出了 CIFAR-10 的一个权重图，且 W 是在训练的过程中不断更新的，因为要确保这些参数能适应所有的训练集。

  ![weight](https://ae01.alicdn.com/kf/Ha83b8bdc7e374e349ece970b0a7bd2f5q.png)

* b 是偏置，也是我们需要训练的参数，相当于一次函数的截距，在线性分类器中相当于平移分隔面，而改变 W 的作用相当于是旋转了分隔面。

  ![bias](https://ae01.alicdn.com/kf/H2531e6abf4574b489d4ae0d138312096N.png)
  
  

* 虽然 W 和 b 表示的是不同的意义，但是我们很多情况下都是将他们两个放在一起的，将 W 和 b 组成一个增广矩阵，这样的话，xi 就多了一个维度，只需要在最后增加一个 `1` ，内积的结果就跟之前一毛一样了，这样子的好处就是我们只需要训练一个矩阵，而不用训练两个矩阵。

  ![w & b](https://ae01.alicdn.com/kf/Hc88d2be5297f4c469c2f399737907d52a.png)



---



对图像而言，每一个像素都是一个特征，特征值在 [0,255] 之间，但是我们通常不这样做，通常要将特征值归一化(normalization)，一种可能的做法是求出训练集中间的一张图片，然后让所有的图片与这张图片相减，得到的特征值大约在 [-127, 127] 间，更加普遍的做法就是让特征值保持在 [-1, 1] 之间。



---



![cat](https://ae01.alicdn.com/kf/H2674833ce66441799c8990abc1b2135bj.png)



假设我们输入一张猫的照片，分类器却跟我们说这是一条狗，那我们就会觉得这个分类器很垃圾，也就是说，W 训练得很垃圾，所以我们要用一些手段来不断地提高 W ，这就要用到 `Loss function` ，也叫 `Cost function` ，通俗点说就是：



```
输入一张猫片
分类器：狗     我：什么辣鸡，继续训练
分类器：还是狗  我：辣鸡，再去训练
分类器：猫     我：牛逼！
```



---



`Loss function` 的种类有很多，其中一种是 `Multiclass SVM loss` ，公式如下(其中 s 是 Score 的缩写)：



```
Li = ∑j≠yi max(0, sj − syi + Δ)
Li = ∑j≠yi max(0, wTjxi − wTyixi + Δ) //推广到向量
```



`SVM loss` 大概的原理就是它有一个超参数 `delta` ，相当于一个 `margin` 吧，SVM 想让正确类别的分数至少比错误类别的分数高 `delta` ，如果分类之后错误的类的 `Score` 与正确类别的 `Score` 的差值在 `delta` 范围内的话就会产生 `Loss` ，否则 `Loss` 就是 0 ，也就是说其他类的得分比正确类的得分少很多。下图很好的解释了这个观点

![svmLoss](https://ae01.alicdn.com/kf/H68dea59c72ef496da41a470f942400fec.png)



---



但是上述的 Loss function 有个 bug， 那就是能让输出结果正确的 W 不止一个，因为线性分类器只需要做矩阵相乘就能够得到各个类别的 Score ， W 乘以一个系数 λ 之后分类结果依然不变，但是 W 乘以一个系数 λ 后 Loss function 的值会改变，原本两个 Score 相减之后只有 10 ，乘以 2 之后差就变成了 20 ，如果 delta 设置的是 15 那么就会产生 Loss 



---



为了消除这种歧义，引入了 正则化 **Regularization** 这个概念，我们要做的就是在上述的 Loss function 中加上正则化惩罚 **Regularization Penalty** , 这里用 **R(W)** 来表示. 最普遍的正则化惩罚函数就是 `L2 范数` ,也就是对 W 的每一个元素都求平方再相加, 公式如下: 



![RW](https://ae01.alicdn.com/kf/H9c4ffa8d97104447a7ebef168a15c434d.png)



注意到我们的正则化惩罚函数只是对 W 进行操作, 并没有对数据进行改动, 因此整一个 Loss function 应该包含两部分, 一部分是 data loss , 也就是所有数据样本的 Loss 的平均值, 再加上正则化惩罚, 公式表达如下:



![total_loss](https://ae01.alicdn.com/kf/H1238ff2bd1c84280a8765c0e70ba07f7d.png)



将公式展开得更具体点就是下面这个 Loss function , 其中 N 就是整个样本的数量, λ 也是一个超参数, 这个值一般是通过交叉验证来确定的. 当然, 老师讲到,在划分训练集的时候就得把标签弄正确,这有助于降低 loss.



![detail](https://ae01.alicdn.com/kf/Hfdfc1be027db4ed0a2991e4d47e86170W.png)



> Additionally, making good predictions on the training set is equivalent to minimizing the loss.



关于上述 delta 的选择方案: 纠结 delta 的具体值是没有意义的，因为 W 可以变大或变小，唯一的权衡因素是我们希望 W 怎样增长（在正则化系数 λ 的作用下）



> 正则化：防止模型过拟合(在训练集上表现太好)，简化模型(权重)，使其在测试集上也能表现很好

![normalization](https://ae01.alicdn.com/kf/Hae5ba152d0564d4fa53e3c11ac1abbbbv.png)