---
layout: post
title: PyTorch JIT 模型部署常见错误
subtitle: 
date: 2021-12-13
author: kevin
header-img: img/green-bg.jpg
catalog: true
tags:
    - troubleshoot
    - linux
    - python
    - model deployment
    - pytorch
    - deep learning

---



## preface



在用 PyTorch官方提供的的工具转化 pth 文件 到 pt 文件时，经常会遇到很多错误，包括但不限于算子不支持，无法推断参数类型，以及一些很奇怪的错误，这里全部记录一下，建议配合我之前写的 MODNet转化模型填坑笔记一起看



## 将 pt 文件保存错位置了



我出现下面这个错误的原因是因为我将模型保存的位置给写错了，所以模型保存失败，解决方法就是换成正确的路径

```
terminate called after throwing an instance of 'c10::Error'
  what():  [enforce fail at inline_container.cc:366] . PytorchStreamWriter failed writing file version: file write failed
frame #0: c10::ThrowEnforceNotMet(char const*, int, char const*, std::string const&, void const*) + 0x47 (0x7f83352836c7 in /raid/kevin/anaconda3/envs/open-mmlab/lib/python3.7/site-packages/torch/lib/libc10.so)
frame #1: caffe2::serialize::PyTorchStreamWriter::valid(char const*, char const*) + 0xa2 (0x7f836d9c8b02 in /raid/kevin/anaconda3/envs/open-mmlab/lib/python3.7/site-packages/torch/lib/libtorch_cpu.so)
frame #2: caffe2::serialize::PyTorchStreamWriter::writeRecord(std::string const&, void const*, unsigned long, bool) + 0x191 (0x7f836d9c9581 in /raid/kevin/anaconda3/envs/open-mmlab/lib/python3.7/site-packages/torch/lib/libtorch_cpu.so)
frame #3: caffe2::serialize::PyTorchStreamWriter::writeEndOfFile() + 0xe1 (0x7f836d9ca101 in /raid/kevin/anaconda3/envs/open-mmlab/lib/python3.7/site-packages/torch/lib/libtorch_cpu.so)
frame #4: caffe2::serialize::PyTorchStreamWriter::~PyTorchStreamWriter() + 0x115 (0x7f836d9ca8f5 in /raid/kevin/anaconda3/envs/open-mmlab/lib/python3.7/site-packages/torch/lib/libtorch_cpu.so)
frame #5: <unknown function> + 0x3554d5d (0x7f836ef31d5d in /raid/kevin/anaconda3/envs/open-mmlab/lib/python3.7/site-packages/torch/lib/libtorch_cpu.so)
frame #6: torch::jit::ExportModule(torch::jit::Module const&, std::string const&, std::unordered_map<std::string, std::string, std::hash<std::string>, std::equal_to<std::string>, std::allocator<std::pair<std::string const, std::string> > > const&, bool) + 0x300 (0x7f836ef310b0 in /raid/kevin/anaconda3/envs/open-mmlab/lib/python3.7/site-packages/torch/lib/libtorch_cpu.so)
frame #7: <unknown function> + 0x67f542 (0x7f8372f26542 in /raid/kevin/anaconda3/envs/open-mmlab/lib/python3.7/site-packages/torch/lib/libtorch_python.so)
frame #8: <unknown function> + 0x2672a8 (0x7f8372b0e2a8 in /raid/kevin/anaconda3/envs/open-mmlab/lib/python3.7/site-packages/torch/lib/libtorch_python.so)
<omitting python frames>
frame #25: __libc_start_main + 0xea (0x7f838e930d0a in /lib/x86_64-linux-gnu/libc.so.6)
```



## 未将模型变成 eval 模式



下面这段错误是因为模型没有变成 eval 模式，导致 JIT 计算出来的结果和预期的结果相差太大，解决方案就是 `your_model.eval()`

```
TracerWarning: Output nr 2. of the traced function does not match the corresponding output of the Python function. Detailed error:
With rtol=1e-05 and atol=1e-05, found 32952 element(s) (out of 128000) whose difference(s) exceeded the margin of error (including 0 nan comparisons). The greatest difference was 0.0001479014754295349 (0.09389541298151016 vs. 0.0940433144569397), which occurred at index (0, 959, 8, 2).
  check_tolerance, strict, _force_outplace, True, _module_class)
```



## reference

[model trace error · Issue #43196 · pytorch/pytorch (github.com)](https://github.com/pytorch/pytorch/issues/43196)
