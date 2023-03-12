---
layout: post
title: ONNX与TensorRT系列
subtitle: 
date: 2022-05-22
author: kevin
header-img: img/green-bg.jpg
catalog: true
tags:
    - linux
    - object detection
    - deep learning
    - mmdetection

---





\## onnx



本质上就是一个有向无环图，用 trace 的方法以一个 dummy tensor 来前向推理一遍网络，来记录下经过的结点，形成一个 graph。



用 `onnx_model.graph.node` 可以得到所有的节点信息，每一个节点里面都有属性，name, input，output，等信息，netron 就是根据这个进行可视化的。



 PyTorch 模型在导出到 ONNX 模型时，模型的输入参数的类型必须全部是 torch.Tensor。而实际上我们传入的第二个参数" 3 "是一个整形变量。这不符合 PyTorch 转 ONNX 的规定。我们必须要修改一下原来的模型的输入。为了保证输入的所有参数都是 torch.Tensor 类型的。



 torch.onnx.export 中需要的模型实际上是一个 torch.jit.ScriptModule。而要把普通 PyTorch 模型转一个这样的 TorchScript 模型，有跟踪（trace）和记录（script）两种导出计算图的方法。如果给 torch.onnx.export 传入了一个普通 PyTorch 模型 (torch.nn.Module)，那么这个模型会默认使用跟踪的方法导出。这一过程如下图所示：





 有些时候，我们希望模型在直接用 PyTorch 推理时有一套逻辑，而在导出的 ONNX 模型中有另一套逻辑。比如，我们可以把一些后处理的逻辑放在模型里，以简化除运行模型之外的其他代码。torch.onnx.is_in_onnx_export() 可以实现这一任务，该函数仅在执行 torch.onnx.export() 时为真。以下是一个例子：



\--- 

 在转换普通的 torch.nn.Module 模型时，PyTorch 一方面会用跟踪法执行前向推理，把遇到的算子整合成计算图；另一方面，PyTorch 还会把遇到的每个算子翻译成 ONNX 中定义的算子。在这个翻译过程中，可能会碰到以下情况：





· 该算子可以一对一地翻译成一个 ONNX 算子。



· 该算子在 ONNX 中没有直接对应的算子，会翻译成一至多个 ONNX 算子。



· 该算子没有定义翻译成 ONNX 的规则，报错。



\---





ONNX 算子的定义情况，都可以在官方的算子文档中查看。这份文档十分重要，我们碰到任何和 ONNX 算子有关的问题都得来”请教“这份文档。





算子文档链接：



https://github.com/onnx/onnx/blob/main/docs/Operators.md



在 PyTorch 中，和 ONNX 有关的定义全部放在 torch.onnx 目录中，如下图所示：





torch.onnx 目录网址：



https://github.com/pytorch/pytorch/tree/master/torch/onnx





使用 torch.onnx.is_in_onnx_export() 来使模型在转换到 ONNX 时有不同的行为.





\--- 



跟踪法得到的 ONNX 模型结构。可以看出来，对于不同的 n，ONNX 模型的结构是不一样的。

而用记录法的话，最终的 ONNX 模型用 Loop 节点来表示循环。这样哪怕对于不同的 n，ONNX 模型也有同样的结构。



\---



在实际的部署过程中，难免碰到模型无法用原生 PyTorch 算子表示的情况。这个时候，我们就得考虑扩充 PyTorch，即在 PyTorch 中支持更多 ONNX 算子。





而要使 PyTorch 算子顺利转换到 ONNX ，我们需要保证以下三个环节都不出错：





· 算子在 PyTorch 中有实现





· 有把该 PyTorch 算子映射成一个或多个 ONNX 算子的方法





· ONNX 有相应的算子





可在实际部署中，这三部分的内容都可能有所缺失。其中最坏的情况是：我们定义了一个全新的算子，它不仅缺少 PyTorch 实现，还缺少 PyTorch 到 ONNX 的映射关系。但所谓车到山前必有路，对于这三个环节，我们也分别都有以下的添加支持的方法：





· PyTorch 算子



\- 组合现有算子



\- 添加 TorchScript 算子



\- 添加普通 C++ 拓展算子





· 映射方法



\- 为 ATen 算子添加符号函数



\- 为 TorchScript 算子添加符号函数



\- 封装成 torch.autograd.Function 并添加符号函数





· ONNX 算子



\- 使用现有 ONNX 算子



\- 定义新 ONNX 算子



---



一般转完 onnx 之后会用 onnxruntime 推理一下进行验证



## TRT 推理遇到的坑



1. pycuda 安装失败	
   1. 源码编译就行了





## 量化三问



1) 为什么量化有用？

因为CNN对噪声不敏感。

2) 为什么用量化？

模型太大，比如alexnet就200MB，贼大，存储压力太大啦；每个层的weights范围基本都是确定的，且波动不大。而且减少访存减少计算量，优势很大的啊！

3) 为什么不直接训练低精度的模型？

因为你训练是需要反向传播和梯度下降的，int8就非常不好做了，举个例子就是我们的学习率一般都是零点几零点几的,你一个int8怎么玩？其次大家的生态就是浮点模型，因此直接转换有效的多啊！



> [(35条消息) 基于tensorRT方案的INT8量化实现原理_alex1801的博客-CSDN博客_tensorrt量化原理](https://blog.csdn.net/weixin_34910922/article/details/108502449)