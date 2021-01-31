---
layout: post
title: ubuntu安装mmdetection
subtitle: 
date: 2020-07-18
author: kevin
header-img: img/green-bg.jpg
catalog: true
tags:
    - linux
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



mmdetection 里面分了好多目录，将相关的文件都放在了同一个文件夹中，下面就会介绍一些重要的文件夹



```txt
├── configs
├── data
├── demo
├── docker
├── docs
├── mmdet
├── mmdet.egg-info
├── requirements
├── resources
├── tests
├── tools
└── work_dirs
```



### configs



```txt
├── albu_example
├── atss
├── _base_
├── carafe
├── cascade_rcnn
├── centripetalnet
├── cityscapes
├── cornernet
├── ………………
├── dcn
├── deepfashion
```



与训练和测试有关的配置都在 `configs` 文件夹中，找到对应算法的 config 文件，config 的命名格式如下

```txt
{model}_[model setting]_{backbone}_{neck}_[norm setting]_[misc]_[gpu x batch_per_gpu]_{schedule}_{dataset}
```

对应的解释如下：

![explain](https://i.loli.net/2020/09/21/cg3vTQHyLW5zjYm.png)



### mmdet



mmdet 这个文件夹是非常重要的，算法源码几乎都在里面，所以重点要知道里面每个文件夹里有什么东西

```txt
├── apis
├── core
├── datasets
├── __init__.py
├── models
├── ops
├── __pycache__
├── utils
└── version.py
```



#### core



core 里面是针对大多数任务都会有的核心操作，比如 anchor 的生成和分配，mAP 计算，bbox 的编码解码，后处理 nms 等等

```txt
├── anchor
├── bbox
├── evaluation
├── export
├── fp16
├── __init__.py
├── mask
├── post_processing
├── __pycache__
└── utils
```



#### datasets



datasets 里面包含了对数据集的处理，尤其是 pipelines 这个文件夹，里面集成了很多个类用来表示对数据集的一些通用操作，如 pad，randomflip，resize 等等，如果想定义一个新的数据集，就得在这里新建一个 py 文件，并且在 `__init__.py` 里面添加这个文件以注册



```txt
├── builder.py
├── cityscapes.py
├── coco.py
├── custom.py
├── dataset_wrappers.py
├── deepfashion.py
├── __init__.py
├── lvis.py
├── pipelines
├── __pycache__
├── samplers
├── utils.py
├── voc.py
├── wider_face.py
└── xml_style.py
```



#### models



models 这个文件夹更加重要，可以说是对我们来说整个 mmdetection 最需要认真看的地方，里面全是一些实现的细节。backbone 里面是分类骨干网络的实现，xx_heads 是网络头部的实现，neck 是连接 backbone 和 heads 的部分，而 detector 里面就是某一个具体的算法的配置，loss 里面实现了各种损失函数。这些都是部件，在 config 的配置中就可以将这些东西组成在一起形成一个具体的算法

```txt
├── backbones
├── builder.py
├── dense_heads
├── detectors
├── __init__.py
├── losses
├── necks
├── __pycache__
├── roi_heads
└── utils
```



### tools



```txt
├── analyze_logs.py
├── benchmark.py
├── browse_dataset.py
├── coco_error_analysis.py
├── convert_datasets
├── detectron2pytorch.py
├── dist_test.sh
├── dist_train.sh
├── eval_metric.py
├── get_flops.py
├── print_config.py
├── ………………
├── train.py
└── upgrade_model_version.py
```



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

## 修改代码



讲一下我在修改 fcos 的过程中学习到的 mmdetection 处理流程以及相应代码的解释

// TODO





## reference



https://mmdetection.readthedocs.io/en/latest

