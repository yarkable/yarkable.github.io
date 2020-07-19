---
layout: post
title: ubuntu安装mmdetection
subtitle: 
date: 2020-07-18
author: kevin
header-img: img/green-bg.jpg
catalog: true
tags:
    - Linux
    - object detection
    - deep learning
---





## 安装



1 先创建一个名为 open-mmlab 的虚拟环境并激活

```bash
$ conda create -n open-mmlab python=3.7 -y
$ conda activate open-mmlab
```

2 安装合适版本的 pytorch（去[官网](https://pytorch.org/)按照自己的 cuda 版本进行安装）

```bash
$ conda install -c pytorch pytorch torchvision -y
```

3 安装 mmcv

```bash
$ pip install mmcv-full==latest+torch1.5.0+cu101 -f https://openmmlab.oss-accelerate.aliyuncs.com/mmcv/dist/index.html
```

(或者直接用下面这条命令也可以)

```bash
$ pip install mmcv-full
```

4 克隆 mmdetection 仓库并进入

```bash
$ git clone https://github.com/open-mmlab/mmdetection.git
$ cd mmdetection
```

5 安装依赖以及 mmdetection

```bash
$ pip install -r requirements/build.txt
$ pip install -v -e .  # or "python setup.py develop"
```



## 使用



//TODO



## reference



https://mmdetection.readthedocs.io/en/latest

