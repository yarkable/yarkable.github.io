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

3 安装 mmcv（看清楚自己的 CUDA 版本）

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



### configs



与训练和测试有关的配置都在 `configs` 文件夹中，找到对应算法的 config 文件，config 的命名格式如下

```txt
{model}_[model setting]_{backbone}_{neck}_[norm setting]_[misc]_[gpu x batch_per_gpu]_{schedule}_{dataset}
```

对应的解释如下：

![explain](https://i.loli.net/2020/09/21/cg3vTQHyLW5zjYm.png)



### models



### tools



#### train



在 `tools` 文件夹里有很多有用的工具，比如说训练测试以及可视化曲线、计算模型参数量的代码。用得十分频繁，下面拿训练 maskrcnn 来做个例子，训练的时候我们就可以用下面命令

```python
CUDA_VISIBLE_DEVICES=2 python tools/train.py configs/mask_rcnn/mask_rcnn_r50_fpn_1x_coco.py
```

mmdetection 就会按照 py 文件中的配置找到相对应的模型配置、训练策略配置以及数据集配置，然后就会开始训练，训练的日志以及模型保存在 `work_dirs` 这个文件夹中。



![option_train](https://i.loli.net/2020/10/08/wufJaWKA72X9dop.png)

#### test



测试的时候就可以用到 `tools/test.py` 来测试模型的 mAP 等等，以及可视化样本，上面训练完成 maskrcnn 之后就可以通过下面的命令来测试性能，这里我是创建了一个新的文件夹 `show_dirs` 用来保存处理后的图片样本

```python
CUDA_VISIBLE_DEVICES=2 python tools/test.py configs/mask_rcnn/mask_rcnn_r50_fpn_1x_coco.py work_dirs/mask_rcnn_r50_fpn_1x_coco/latest.pth --show-dir show_dirs/maskrcnn --eval segm
```

![sample](https://i.loli.net/2020/10/08/SzZV437hsaQnowg.png)



全部的测试选项代码如下

```python
python tools/test.py ${CONFIG_FILE} ${CHECKPOINT_FILE} [--out ${RESULT_FILE}] [--eval ${EVAL_METRICS}] [--show] [--cfg-options]
```

![option_test](https://i.loli.net/2020/10/08/o5Ia3drxK29cZBW.png)



#### analyze_logs



然后我们可以用训练的日志文件来生成 loss 曲线等，这个在 `tools/analyze_logs.py` 文件中有提及，挺方便的

```python
CUDA_VISIBLE_DEVICES=2 python tools/analyze_logs.py  plot_curve work_dirs/mask_rcnn_r50_fpn_1x_coco/20200912_024022.log.json --keys loss_bbox loss_mask --legend loss_bbox loss_mask --title LOSS_CURVE --out loss.jpg
```



然后就会得到这样的一张 loss 曲线图片

![loss.jpg](https://i.loli.net/2020/10/08/sAcP6mGRC5EYKXZ.jpg)



全部的代码选项如下

```python
python tools/analyze_logs.py plot_curve [--keys ${KEYS}] [--title ${TITLE}] [--legend ${LEGEND}] [--backend ${BACKEND}] [--style ${STYLE}] [--out ${OUT_FILE}]
```

![option_analyze_logs](https://i.loli.net/2020/10/08/3dvQpZI5aTnPybx.png)



## reference



https://mmdetection.readthedocs.io/en/latest

