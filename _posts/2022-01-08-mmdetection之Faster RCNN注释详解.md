---
layout: post
title: mmdetection之Faster RCNN注释详解
subtitle: 
date: 2022-01-08
author: kevin
header-img: img/green-bg.jpg
catalog: true
tags:
    - linux
    - object detection
    - deep learning
    - mmdetection
---



## preface 



本文记录 mmdetection 对 Faster RCNN 训练的流程，包括标签获取，anchor 生成，前向训练，以及各步骤中 tensor 的形状，仅供复习用处。mmdetection 版本为 2.11.0。



## 整体流程



Faster RCNN 作为二阶段检测器，第一阶段在特征图上生成密集的 anchor，通过简单的卷积筛选掉一些置信度很低的 anchor，并且控制正负样本 anchor 的比例，将这些 anchor 以及对应的特征送入第二阶段进行精细的分类和回归，第一阶段就相当是做了一个二分类。



1. 图片输入到 ResNet 中进行特征提取，输出 4 个特征图，按照特征图从大到小排列，分别是 C2 C3 C4 C5，stride = 4,8,16,32
2. 4 个特征图输入到 FPN 模块中进行特征融合，输出 5 个通道数相同的特征图,分别是 p2 ~ p6，stride = 4,8,16,32,64
3. FPN 输出的 5 个特征图，输入到同一个 RPN 或者说 5 个相同的 RPN 中，每个分支都进行前后景分类和 bbox 回归，然后就可以和 label 计算 loss
4. 在 5 个 RPN 分支输出的基础上，采用 RPN test 模块输出指定个数的 Region Proposal，将 Region Proposal 按照重映射规则，在对应的 p2 ~ p5 特征图上进行特征提取，注意并没有使用 p6 层特征图，从而得到指定个数例如 2k 个 Region Proposal 特征图
5. 将 2k 个不同大小的 RoI 区域特征图输入到 RoIAlign 或者 RoIPool 层中进行统一采样，得到指定输出 shape 的 2k 个特征图
6. 组成 batch 输入到两层 FC 中进行多类别的分类和回归，其 loss 和 RPN 层 loss 相加进行联合训练





## Faster RCNN



先从整个模型的 detector 看起，Faster RCNN 直接继承了 `TwoStageDetector`，没有做出什么改动，所以直接去看 `TwoStageDetector` 里面的内容就行了

```python
@DETECTORS.register_module()
class FasterRCNN(TwoStageDetector):
    """Implementation of `Faster R-CNN <https://arxiv.org/abs/1506.01497>`_"""

    def __init__(self,
                 backbone,
                 rpn_head,
                 roi_head,
                 train_cfg,
                 test_cfg,
                 neck=None,
                 pretrained=None):
        super(FasterRCNN, self).__init__(
            backbone=backbone,
            neck=neck,
            rpn_head=rpn_head,
            roi_head=roi_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            pretrained=pretrained)
```



## TwoStageDetector.forward_train



整体的流程还是很清晰的，首先通过 FPN 得到不同尺度的特征图，然后通过 RPN 给 anchor 分类，筛选一部分正负样本训练，然后根据这些样本的坐标去特征图中取出对应的特征进行第二阶段的分类和回归。

```python
def forward_train(self,
                  img,
                  img_metas,
                  gt_bboxes,
                  gt_labels,
                  gt_bboxes_ignore=None,
                  gt_masks=None,
                  proposals=None,
                  **kwargs):

    x = self.extract_feat(img)

    losses = dict()

    # Faster 是有 RPN 的，Fast RCNN 才没有，用的 selective search 得到的 proposal
    if self.with_rpn:
        proposal_cfg = self.train_cfg.get('rpn_proposal',
                                          self.test_cfg.rpn)
        # RPN 前向得到 rpn_loss 和 proposals 框
        rpn_losses, proposal_list = self.rpn_head.forward_train(
            x,
            img_metas,
            gt_bboxes,
            gt_labels=None,
            gt_bboxes_ignore=gt_bboxes_ignore,
            proposal_cfg=proposal_cfg)
        losses.update(rpn_losses)
    else:
        proposal_list = proposals
	
    # ROI_Head 前向传播得到分类和回归 loss
    roi_losses = self.roi_head.forward_train(x, img_metas, proposal_list,
                                             gt_bboxes, gt_labels,
                                             gt_bboxes_ignore, gt_masks,
                                             **kwargs)
    losses.update(roi_losses)

    return losses
```



## RPNHead



介绍完流程后，首先就来介绍 RPNHead，这个类继承了 `RPNTestMixin` 和 `AnchorHead` 两个类，第一个类是用来得到 RPN 前向传播出的 proposal 的， 第二个类就是一个带 Anchor 的检测头的集成算法。

```python
@HEADS.register_module()
class RPNHead(RPNTestMixin, AnchorHead):
    """RPN head.

    Args:
        in_channels (int): Number of channels in the input feature map.
    """  # noqa: W605

    def __init__(self, in_channels, **kwargs):
        # 初始化 self.num_classes 为1
        super(RPNHead, self).__init__(1, in_channels, **kwargs)

    def _init_layers(self):
        """Initialize layers of the head."""
        # 只用一层卷积进行二分类
        self.rpn_conv = nn.Conv2d(
            self.in_channels, self.feat_channels, 3, padding=1)
        # self.cls_out_channels 为1，因为是二分类
        # 并且 config 里面使用了 use_sigmoid=True，所以对 num_classes 不会加 1 
        self.rpn_cls = nn.Conv2d(self.feat_channels,
                                 self.num_anchors * self.cls_out_channels, 1)
        self.rpn_reg = nn.Conv2d(self.feat_channels, self.num_anchors * 4, 1)

    def init_weights(self):
        """Initialize weights of the head."""
        normal_init(self.rpn_conv, std=0.01)
        normal_init(self.rpn_cls, std=0.01)
        normal_init(self.rpn_reg, std=0.01)
```



Faster RCNN config 里面 rpn_head 的 loss_cls 用了 use_sigmoid=True，对 anchor 进行分类

```python
loss_cls=dict(
    type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0),
```

anchor_head 里面对 self.cls_out_channels 进一步判断

```python
if self.use_sigmoid_cls:
    self.cls_out_channels = num_classes
else:
    self.cls_out_channels = num_classes + 1
```



### RPNHead.forward_train



RPNHead 里面并没有实现 forward_train，所以是继承了 anchor_head 中的 forward_train，但是 anchor_head 也没有实现这个方法，因此是继承 base_dense_head 中的 forward_train，如下。

```python
def forward_train(self,
                  x,
                  img_metas,
                  gt_bboxes,
                  gt_labels=None,
                  gt_bboxes_ignore=None,
                  proposal_cfg=None,
                  **kwargs):
    """
    Args:
        x (list[Tensor]): Features from FPN.
        img_metas (list[dict]): Meta information of each image, e.g.,
            image size, scaling factor, etc.
        gt_bboxes (Tensor): Ground truth bboxes of the image,
            shape (num_gts, 4).
        gt_labels (Tensor): Ground truth labels of each box,
            shape (num_gts,).
        gt_bboxes_ignore (Tensor): Ground truth bboxes to be
            ignored, shape (num_ignored_gts, 4).
        proposal_cfg (mmcv.Config): Test / postprocessing configuration,
            if None, test_cfg would be used

    Returns:
        tuple:
            losses: (dict[str, Tensor]): A dictionary of loss components.
            proposal_list (list[Tensor]): Proposals of each image.
    """    
    outs = self(x)
    # 进行而分类，gt_label=None
    if gt_labels is None:
        loss_inputs = outs + (gt_bboxes, img_metas)
    else:
        loss_inputs = outs + (gt_bboxes, gt_labels, img_metas)
    losses = self.loss(*loss_inputs, gt_bboxes_ignore=gt_bboxes_ignore)
    if proposal_cfg is None:
        return losses
    else:
        proposal_list = self.get_bboxes(*outs, img_metas, cfg=proposal_cfg)
        return losses, proposal_list
```



self(x) 调用了 base_dense_head 的 forward 函数，他没有实现，所以调用 anchor_head 的 forward，然后 rpn_head 又重写了 forward_single 这个函数，所以最终是通过 rpn_head 的 forward_single 函数将每一个 FPN 后的特征图通过一个共享的卷积和两个分类回归分支得到输出。

```python
def forward_single(self, x):
    """Forward feature map of a single scale level."""
    x = self.rpn_conv(x)
    x = F.relu(x, inplace=True)
    rpn_cls_score = self.rpn_cls(x)
    rpn_bbox_pred = self.rpn_reg(x)
    return rpn_cls_score, rpn_bbox_pred
```



gt_label=None 是因为第一阶段只是对 anchor 进行正样本或负样本的分类，。紧接着就开始计算 rpn 阶段的 loss。



### RPNHead.loss



rpn_loss 在这里也只是调用的 anchor_head 的 loss 函数，只不过将 gt_label=None 传入函数。

```python
def loss(self,
         cls_scores,
         bbox_preds,
         gt_bboxes,
         img_metas,
         gt_bboxes_ignore=None):
    """Compute losses of the head.

    Args:
        cls_scores (list[Tensor]): Box scores for each scale level
            Has shape (N, num_anchors * num_classes, H, W)
        bbox_preds (list[Tensor]): Box energies / deltas for each scale
            level with shape (N, num_anchors * 4, H, W)
        gt_bboxes (list[Tensor]): Ground truth bboxes for each image with
            shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
        img_metas (list[dict]): Meta information of each image, e.g.,
            image size, scaling factor, etc.
        gt_bboxes_ignore (None | list[Tensor]): specify which bounding
            boxes can be ignored when computing the loss.

    Returns:
        dict[str, Tensor]: A dictionary of loss components.
    """
    # 调用 anchor_head 的 loss 函数得到 box loss 和二分类 loss
    losses = super(RPNHead, self).loss(
        cls_scores,
        bbox_preds,
        gt_bboxes,
        None,
        img_metas,
        gt_bboxes_ignore=gt_bboxes_ignore)
    return dict(
        loss_rpn_cls=losses['loss_cls'], loss_rpn_bbox=losses['loss_bbox'])
```



### Anchor_head._get_targets_single



所以就和普通的 anchor_based 的算法一样，首先在每个特征图上都生成一堆 anchor，然后将所有 anchor 和所有 gt 一一匹配，互相找到 IoU 最大的索引，然后确定回归和分类目标，只不过分类目标是 0 或者 1，0代表正样本，1 代表负样本，在 anchor_head 中有写道

```python
if gt_labels is None:
    # Only rpn gives gt_labels as None
    # Foreground is the first class since v2.5.0
    labels[pos_inds] = 0
```

并且由于 Faster RCNN 是二阶段的算法，第一阶段会筛选掉大量的没用的负样本，并且控制正负样本的比例，所以还会涉及到 sampler 取样器，



```python
def _get_targets_single(self,
                        flat_anchors,
                        valid_flags,
                        gt_bboxes,
                        gt_bboxes_ignore,
                        gt_labels,
                        img_meta,
                        label_channels=1,
                        unmap_outputs=True):

    inside_flags = anchor_inside_flags(flat_anchors, valid_flags,
                                        img_meta['img_shape'][:2],
                                        self.train_cfg.allowed_border)
    if not inside_flags.any():
        return (None, ) * 7
    # assign gt and sample anchors
    # torch.Size([217413, 4])
    anchors = flat_anchors[inside_flags, :]
	
  	# 对 anchor 进行标签分配
    assign_result = self.assigner.assign(
        anchors, gt_bboxes, gt_bboxes_ignore,
        None if self.sampling else gt_labels)
    # 进行取样，筛选一部分正负样本进行训练
    sampling_result = self.sampler.sample(assign_result, anchors,
                                            gt_bboxes)

    num_valid_anchors = anchors.shape[0]
    bbox_targets = torch.zeros_like(anchors)
    bbox_weights = torch.zeros_like(anchors)
    labels = anchors.new_full((num_valid_anchors, ),
                                self.num_classes,
                                dtype=torch.long)
    label_weights = anchors.new_zeros(num_valid_anchors, dtype=torch.float)
	
  	# shape: 72
    pos_inds = sampling_result.pos_inds
    # shape: 184，加起来刚好是 256，在下文 Sampler 中会讲到
    neg_inds = sampling_result.neg_inds
    if len(pos_inds) > 0:
        if not self.reg_decoded_bbox:
            pos_bbox_targets = self.bbox_coder.encode(
                sampling_result.pos_bboxes, sampling_result.pos_gt_bboxes)
        else:
            pos_bbox_targets = sampling_result.pos_gt_bboxes
        bbox_targets[pos_inds, :] = pos_bbox_targets
        bbox_weights[pos_inds, :] = 1.0
        if gt_labels is None:
            # Only rpn gives gt_labels as None
            # Foreground is the first class since v2.5.0
            # 第一阶段二分类，正样本的标签被设定为 0，背景的标签是 1
            labels[pos_inds] = 0
        else:
            labels[pos_inds] = gt_labels[
                sampling_result.pos_assigned_gt_inds]
        if self.train_cfg.pos_weight <= 0:
            label_weights[pos_inds] = 1.0
        else:
            label_weights[pos_inds] = self.train_cfg.pos_weight
    if len(neg_inds) > 0:
        label_weights[neg_inds] = 1.0

```



### RPNHead.train_cfg.assigner



正负样本匹配的策略，跟其他的 anchor_base 的算法差不多，Faster RCNN 设定与 gt_box 的 IoU 大于 0.7 为正样本，小于 0.3 为负样本，中间的为忽略样本。



```python
self.train_cfg = train_cfg
self.test_cfg = test_cfg
if self.train_cfg:
    self.assigner = build_assigner(self.train_cfg.assigner)
    # use PseudoSampler when sampling is False
    if self.sampling and hasattr(self.train_cfg, 'sampler'):
        sampler_cfg = self.train_cfg.sampler
    else:
        sampler_cfg = dict(type='PseudoSampler')
    self.sampler = build_sampler(sampler_cfg, context=self)
```

```python
assigner=dict(
    type='MaxIoUAssigner',
    pos_iou_thr=0.7,
    neg_iou_thr=0.3,
    min_pos_iou=0.3,
    match_low_quality=True,
    ignore_iof_thr=-1),
```



### RPNHead.train_cfg.sampler



Faster RCNN 算法用的是 `RandomSampler`，对分配好标签的 anchor 进行随机取样，先看看 config 里面关于 sampler 的配置：

```python
sampler=dict(
    type='RandomSampler',
  	# 总共拿去训练的样本为 256 个
    num=256,
  	# 正样本占总数的一半，不够的话就用负样本代替
    pos_fraction=0.5,
  	# 负样本与正样本比值的最大值，默认是 -1
    neg_pos_ub=-1,
  	# 将 gt_box 认定为一个 proposal，默认是 True
    add_gt_as_proposals=False)
```

分析一下源码：

```python
@BBOX_SAMPLERS.register_module()
class RandomSampler(BaseSampler):
    """Random sampler.

    Args:
        num (int): Number of samples
        pos_fraction (float): Fraction of positive samples
        neg_pos_up (int, optional): Upper bound number of negative and
            positive samples. Defaults to -1.
        add_gt_as_proposals (bool, optional): Whether to add ground truth
            boxes as proposals. Defaults to True.
    """

    def __init__(self,
                 num,
                 pos_fraction,
                 neg_pos_ub=-1,
                 add_gt_as_proposals=True,
                 **kwargs):
        from mmdet.core.bbox import demodata
        super(RandomSampler, self).__init__(num, pos_fraction, neg_pos_ub,
                                            add_gt_as_proposals)
        self.rng = demodata.ensure_rng(kwargs.get('rng', None))

    def random_choice(self, gallery, num):
        """Random select some elements from the gallery.

        If `gallery` is a Tensor, the returned indices will be a Tensor;
        If `gallery` is a ndarray or list, the returned indices will be a
        ndarray.

        Args:
            gallery (Tensor | ndarray | list): indices pool.
            num (int): expected sample num.

        Returns:
            Tensor or ndarray: sampled indices.
        """
        assert len(gallery) >= num

        is_tensor = isinstance(gallery, torch.Tensor)
        if not is_tensor:
            if torch.cuda.is_available():
                device = torch.cuda.current_device()
            else:
                device = 'cpu'
            gallery = torch.tensor(gallery, dtype=torch.long, device=device)
        # 有用的只有这一句，先将 gallery 中的元素随机打乱，再取前 num 个元素，达到目的
        perm = torch.randperm(gallery.numel(), device=gallery.device)[:num]
        rand_inds = gallery[perm]
        if not is_tensor:
            rand_inds = rand_inds.cpu().numpy()
        return rand_inds

    def _sample_pos(self, assign_result, num_expected, **kwargs):
        """Randomly sample some positive samples."""
        # 得到了正样本的索引
        pos_inds = torch.nonzero(assign_result.gt_inds > 0, as_tuple=False)
        if pos_inds.numel() != 0:
            pos_inds = pos_inds.squeeze(1)
        # 正样本不够的话就直接返回
        if pos_inds.numel() <= num_expected:
            return pos_inds
        else:
          	# 对正样本随机取样
            return self.random_choice(pos_inds, num_expected)

    def _sample_neg(self, assign_result, num_expected, **kwargs):
        """Randomly sample some negative samples."""
        # 同林
        neg_inds = torch.nonzero(assign_result.gt_inds == 0, as_tuple=False)
        if neg_inds.numel() != 0:
            neg_inds = neg_inds.squeeze(1)
        if len(neg_inds) <= num_expected:
            return neg_inds
        else:
            return self.random_choice(neg_inds, num_expected)

```

函数调用的是 `self.sampler.sample`，这里没有实现，实现在基类中，接下去看一下。

### BaseSampler



这是取样器的基类，包含了取样的过程，分析一下，主要是 sample 函数。

```python
class BaseSampler(metaclass=ABCMeta):
    """Base class of samplers."""

    def __init__(self,
                 num,
                 pos_fraction,
                 neg_pos_ub=-1,
                 add_gt_as_proposals=True,
                 **kwargs):
        self.num = num
        self.pos_fraction = pos_fraction
        self.neg_pos_ub = neg_pos_ub
        self.add_gt_as_proposals = add_gt_as_proposals
        # 好骚的写法
        self.pos_sampler = self
        self.neg_sampler = self

    @abstractmethod
    def _sample_pos(self, assign_result, num_expected, **kwargs):
        """Sample positive samples."""
        pass

    @abstractmethod
    def _sample_neg(self, assign_result, num_expected, **kwargs):
        """Sample negative samples."""
        pass

    def sample(self,
               assign_result,
               bboxes,
               gt_bboxes,
               gt_labels=None,
               **kwargs):
        """Sample positive and negative bboxes.

        This is a simple implementation of bbox sampling given candidates,
        assigning results and ground truth bboxes.

        Args:
            assign_result (:obj:`AssignResult`): Bbox assigning results.
            bboxes (Tensor): Boxes to be sampled from.
            gt_bboxes (Tensor): Ground truth bboxes.
            gt_labels (Tensor, optional): Class labels of ground truth bboxes.

        Returns:
            :obj:`SamplingResult`: Sampling result.

        Example:
            >>> from mmdet.core.bbox import RandomSampler
            >>> from mmdet.core.bbox import AssignResult
            >>> from mmdet.core.bbox.demodata import ensure_rng, random_boxes
            >>> rng = ensure_rng(None)
            >>> assign_result = AssignResult.random(rng=rng)
            >>> bboxes = random_boxes(assign_result.num_preds, rng=rng)
            >>> gt_bboxes = random_boxes(assign_result.num_gts, rng=rng)
            >>> gt_labels = None
            >>> self = RandomSampler(num=32, pos_fraction=0.5, neg_pos_ub=-1,
            >>>                      add_gt_as_proposals=False)
            >>> self = self.sample(assign_result, bboxes, gt_bboxes, gt_labels)
        """
        if len(bboxes.shape) < 2:
            bboxes = bboxes[None, :]

        bboxes = bboxes[:, :4]
				# 表明该 box 是不是 gt 的 flag
        gt_flags = bboxes.new_zeros((bboxes.shape[0], ), dtype=torch.uint8)
        # 前期训练时正样本可能会很少，将 gt 添加进来作为 proposal
        if self.add_gt_as_proposals and len(gt_bboxes) > 0:
            if gt_labels is None:
                raise ValueError(
                    'gt_labels must be given when add_gt_as_proposals is True')
            bboxes = torch.cat([gt_bboxes, bboxes], dim=0)
            assign_result.add_gt_(gt_labels)
            gt_ones = bboxes.new_ones(gt_bboxes.shape[0], dtype=torch.uint8)
            gt_flags = torch.cat([gt_ones, gt_flags])

        num_expected_pos = int(self.num * self.pos_fraction)
        # 对正样本进行取样
        pos_inds = self.pos_sampler._sample_pos(
            assign_result, num_expected_pos, bboxes=bboxes, **kwargs)
        # We found that sampled indices have duplicated items occasionally.
        # (may be a bug of PyTorch)
        pos_inds = pos_inds.unique()
        num_sampled_pos = pos_inds.numel()
        num_expected_neg = self.num - num_sampled_pos
        if self.neg_pos_ub >= 0:
            _pos = max(1, num_sampled_pos)
            neg_upper_bound = int(self.neg_pos_ub * _pos)
            if num_expected_neg > neg_upper_bound:
                num_expected_neg = neg_upper_bound
        # 对负样本进行取样
        neg_inds = self.neg_sampler._sample_neg(
            assign_result, num_expected_neg, bboxes=bboxes, **kwargs)
        neg_inds = neg_inds.unique()

        sampling_result = SamplingResult(pos_inds, neg_inds, bboxes, gt_bboxes,
                                         assign_result, gt_flags)
        return sampling_result

```



到这里，我们完成了第一阶段 RPN 的训练，和 FocalLoss 通过改变 loss 对样本的惩罚不一样，Faster RCNN 使用了随机取样来减少正负样本不平衡带来的训练问题。值得注意的是，第一阶段取样了 256 个正负样本进行训练，但是流程和没有取样是一样的，sample 之后是通过改变 label_weights_list 来取样的，送入后续的 CE 和 L1 loss，里面的值为 1 说明是需要计算 loss 的样本，为 0 说明是不需要计算 loss 的，达到屏蔽的作用，`(label_weights_list[0]==1).sum()` 是 256。

那么，RPN 是如何提取 proposal 送到第二阶段去进行训练的呢，这就要用到 RPNHead 的 test 阶段了，下面介绍。



### RPNHead.get_bboxes



在 `two_stage.py` 中有这么一句，如果配置中有 RPN 的话就掉用 forward_train 得到 RPN loss 和 proposal_list，现在我们得到了 RPN loss 了，我们继续去看看 proposal_list 怎么获得

```python
if self.with_rpn:
  proposal_cfg = self.train_cfg.get('rpn_proposal',
                                    self.test_cfg.rpn)
  # RPN 前向得到 rpn_loss 和 proposals 框
  rpn_losses, proposal_list = self.rpn_head.forward_train(
    x,
    img_metas,
    gt_bboxes,
    gt_labels=None,
    gt_bboxes_ignore=gt_bboxes_ignore,
    proposal_cfg=proposal_cfg)
  losses.update(rpn_losses)
else:
    proposal_list = proposals
```



在 `base_dense_head.py` 中写了下面代码，我么可以看到，获得 proposal 是通过 head 的 get_bboxes 函数的，也就是测试的阶段。

```python
outs = self(x)
if gt_labels is None:
    loss_inputs = outs + (gt_bboxes, img_metas)
else:
    loss_inputs = outs + (gt_bboxes, gt_labels, img_metas)
losses = self.loss(*loss_inputs, gt_bboxes_ignore=gt_bboxes_ignore)
if proposal_cfg is None:
    return losses
else:
    proposal_list = self.get_bboxes(*outs, img_metas, cfg=proposal_cfg)
    return losses, proposal_list
```

这里我们的 head 是 RPNHead，但是 RPNHead 没有实现 get_bboxes 函数，只实现了 _get_bboxes 函数，所以这里是继承的 Anchor_head 的 get_bboxes 函数。这个函数比较通用，其实也就是设置一些东西，主要的逻辑都在 _get_bboxes 里面。



```python
@force_fp32(apply_to=('cls_scores', 'bbox_preds'))
def get_bboxes(self,
               cls_scores,
               bbox_preds,
               img_metas,
               cfg=None,
               rescale=False,
               with_nms=True):
    """Transform network output for a batch into bbox predictions.

    Args:
        cls_scores (list[Tensor]): Box scores for each scale level
            Has shape (N, num_anchors * num_classes, H, W)
        bbox_preds (list[Tensor]): Box energies / deltas for each scale
            level with shape (N, num_anchors * 4, H, W)
        img_metas (list[dict]): Meta information of each image, e.g.,
            image size, scaling factor, etc.
        cfg (mmcv.Config | None): Test / postprocessing configuration,
            if None, test_cfg would be used
        rescale (bool): If True, return boxes in original image space.
            Default: False.
        with_nms (bool): If True, do nms before return boxes.
            Default: True.

    Returns:
        list[tuple[Tensor, Tensor]]: Each item in result_list is 2-tuple.
            The first item is an (n, 5) tensor, where 5 represent
            (tl_x, tl_y, br_x, br_y, score) and the score between 0 and 1.
            The shape of the second tensor in the tuple is (n,), and
            each element represents the class label of the corresponding
            box.

    Example:
        >>> import mmcv
        >>> self = AnchorHead(
        >>>     num_classes=9,
        >>>     in_channels=1,
        >>>     anchor_generator=dict(
        >>>         type='AnchorGenerator',
        >>>         scales=[8],
        >>>         ratios=[0.5, 1.0, 2.0],
        >>>         strides=[4,]))
        >>> img_metas = [{'img_shape': (32, 32, 3), 'scale_factor': 1}]
        >>> cfg = mmcv.Config(dict(
        >>>     score_thr=0.00,
        >>>     nms=dict(type='nms', iou_thr=1.0),
        >>>     max_per_img=10))
        >>> feat = torch.rand(1, 1, 3, 3)
        >>> cls_score, bbox_pred = self.forward_single(feat)
        >>> # note the input lists are over different levels, not images
        >>> cls_scores, bbox_preds = [cls_score], [bbox_pred]
        >>> result_list = self.get_bboxes(cls_scores, bbox_preds,
        >>>                               img_metas, cfg)
        >>> det_bboxes, det_labels = result_list[0]
        >>> assert len(result_list) == 1
        >>> assert det_bboxes.shape[1] == 5
        >>> assert len(det_bboxes) == len(det_labels) == cfg.max_per_img
    """
    assert len(cls_scores) == len(bbox_preds)
    num_levels = len(cls_scores)

    device = cls_scores[0].device
    featmap_sizes = [cls_scores[i].shape[-2:] for i in range(num_levels)]
    mlvl_anchors = self.anchor_generator.grid_anchors(
        featmap_sizes, device=device)
		# detach 掉梯度，不回传
    cls_score_list = [cls_scores[i].detach() for i in range(num_levels)]
    bbox_pred_list = [bbox_preds[i].detach() for i in range(num_levels)]

    if torch.onnx.is_in_onnx_export():
        assert len(
            img_metas
        ) == 1, 'Only support one input image while in exporting to ONNX'
        img_shapes = img_metas[0]['img_shape_for_onnx']
    else:
        img_shapes = [
            img_metas[i]['img_shape']
            for i in range(cls_scores[0].shape[0])
        ]
    scale_factors = [
        img_metas[i]['scale_factor'] for i in range(cls_scores[0].shape[0])
    ]

    if with_nms:
        # some heads don't support with_nms argument
        result_list = self._get_bboxes(cls_score_list, bbox_pred_list,
                                       mlvl_anchors, img_shapes,
                                       scale_factors, cfg, rescale)
    else:
        result_list = self._get_bboxes(cls_score_list, bbox_pred_list,
                                       mlvl_anchors, img_shapes,
                                       scale_factors, cfg, rescale,
                                       with_nms)
    return result_list
```



### RPNHead._get_bboxes



这个函数将一个 batch 的预测转化成 bbox 预测，得到每一张图片在第一阶段产生的 1000 个 proposal

```python
def _get_bboxes(self,
                cls_scores,
                bbox_preds,
                mlvl_anchors,
                img_shapes,
                scale_factors,
                cfg,
                rescale=False):
    """Transform outputs for a single batch item into bbox predictions.

    Args:
        cls_scores (list[Tensor]): Box scores for each scale level
            Has shape (N, num_anchors * num_classes, H, W).
        bbox_preds (list[Tensor]): Box energies / deltas for each scale
            level with shape (N, num_anchors * 4, H, W).
        mlvl_anchors (list[Tensor]): Box reference for each scale level
            with shape (num_total_anchors, 4).
        img_shapes (list[tuple[int]]): Shape of the input image,
            (height, width, 3).
        scale_factors (list[ndarray]): Scale factor of the image arange as
            (w_scale, h_scale, w_scale, h_scale).
        cfg (mmcv.Config): Test / postprocessing configuration,
            if None, test_cfg would be used.
        rescale (bool): If True, return boxes in original image space.

    Returns:
        list[tuple[Tensor, Tensor]]: Each item in result_list is 2-tuple.
            The first item is an (n, 5) tensor, where the first 4 columns
            are bounding box positions (tl_x, tl_y, br_x, br_y) and the
            5-th column is a score between 0 and 1. The second item is a
            (n,) tensor where each item is the predicted class labelof the
            corresponding box.
    """
    cfg = self.test_cfg if cfg is None else cfg
    cfg = copy.deepcopy(cfg)
    # bboxes from different level should be independent during NMS,
    # level_ids are used as labels for batched NMS to separate them
    level_ids = []
    mlvl_scores = []
    mlvl_bbox_preds = []
    mlvl_valid_anchors = []
    batch_size = cls_scores[0].shape[0]
    nms_pre_tensor = torch.tensor(
        cfg.nms_pre, device=cls_scores[0].device, dtype=torch.long)
    # 对 FPN 每一层进行操作
    for idx in range(len(cls_scores)):
        rpn_cls_score = cls_scores[idx]
        rpn_bbox_pred = bbox_preds[idx]
        assert rpn_cls_score.size()[-2:] == rpn_bbox_pred.size()[-2:]
        # 进行 reshape，如果 use_sigmoid=True 的话就直接给分类预测加上 sigmoid 函数得到 score
        rpn_cls_score = rpn_cls_score.permute(0, 2, 3, 1)
        # scores -> shape: [B, num_all_anchor_this_lvl]
        if self.use_sigmoid_cls:
            rpn_cls_score = rpn_cls_score.reshape(batch_size, -1)
            scores = rpn_cls_score.sigmoid()
        else:
            rpn_cls_score = rpn_cls_score.reshape(batch_size, -1, 2)
            # We set FG labels to [0, num_class-1] and BG label to
            # num_class in RPN head since mmdet v2.5, which is unified to
            # be consistent with other head since mmdet v2.0. In mmdet v2.0
            # to v2.4 we keep BG label as 0 and FG label as 1 in rpn head.
            scores = rpn_cls_score.softmax(-1)[..., 0]
        rpn_bbox_pred = rpn_bbox_pred.permute(0, 2, 3, 1).reshape(
            batch_size, -1, 4)
        anchors = mlvl_anchors[idx]
        anchors = anchors.expand_as(rpn_bbox_pred)
        if nms_pre_tensor > 0:
            # sort is faster than topk
            # _, topk_inds = scores.topk(cfg.nms_pre)
            # keep topk op for dynamic k in onnx model
            if torch.onnx.is_in_onnx_export():
                # sort op will be converted to TopK in onnx
                # and k<=3480 in TensorRT
                scores_shape = torch._shape_as_tensor(scores)
                nms_pre = torch.where(scores_shape[1] < nms_pre_tensor,
                                      scores_shape[1], nms_pre_tensor)
                _, topk_inds = scores.topk(nms_pre)
                batch_inds = torch.arange(batch_size).view(
                    -1, 1).expand_as(topk_inds)
                scores = scores[batch_inds, topk_inds]
                rpn_bbox_pred = rpn_bbox_pred[batch_inds, topk_inds, :]
                anchors = anchors[batch_inds, topk_inds, :]
						# 每一层最多选出 nms_pre 个框，这里是 2000 个
            elif scores.shape[-1] > cfg.nms_pre:
              	# 根据第一阶段二分类的预测分值来进行排序，选择分值高的 nms_pre 个样本
                ranked_scores, rank_inds = scores.sort(descending=True)
                # shape: [B, nms_pre]
                topk_inds = rank_inds[:, :cfg.nms_pre]
                scores = ranked_scores[:, :cfg.nms_pre]
                # batch_inds 是为了让 rpn_bbox_pred 在取索引的时候对的上 shape
                # shape: [B, nms_pre]
                batch_inds = torch.arange(batch_size).view(
                    -1, 1).expand_as(topk_inds)
                # shape: [B, nms_pre, 4]，下同
                rpn_bbox_pred = rpn_bbox_pred[batch_inds, topk_inds, :]
                anchors = anchors[batch_inds, topk_inds, :]

        mlvl_scores.append(scores)
        mlvl_bbox_preds.append(rpn_bbox_pred)
        mlvl_valid_anchors.append(anchors)
        level_ids.append(
            scores.new_full((
                batch_size,
                scores.size(1),
            ),
                            idx,
                            dtype=torch.long))
		# shape: [B, nms_pre*num_levels]，可能一层没有 nms_pre 这么多，所以实际上会小于这个数
    batch_mlvl_scores = torch.cat(mlvl_scores, dim=1)
    batch_mlvl_anchors = torch.cat(mlvl_valid_anchors, dim=1)
    batch_mlvl_rpn_bbox_pred = torch.cat(mlvl_bbox_preds, dim=1)
    batch_mlvl_proposals = self.bbox_coder.decode(
        batch_mlvl_anchors, batch_mlvl_rpn_bbox_pred, max_shape=img_shapes)
    batch_mlvl_ids = torch.cat(level_ids, dim=1)

    # deprecate arguments warning
    if 'nms' not in cfg or 'max_num' in cfg or 'nms_thr' in cfg:
        warnings.warn(
            'In rpn_proposal or test_cfg, '
            'nms_thr has been moved to a dict named nms as '
            'iou_threshold, max_num has been renamed as max_per_img, '
            'name of original arguments and the way to specify '
            'iou_threshold of NMS will be deprecated.')
    if 'nms' not in cfg:
        cfg.nms = ConfigDict(dict(type='nms', iou_threshold=cfg.nms_thr))
    if 'max_num' in cfg:
        if 'max_per_img' in cfg:
            assert cfg.max_num == cfg.max_per_img, f'You ' \
                f'set max_num and ' \
                f'max_per_img at the same time, but get {cfg.max_num} ' \
                f'and {cfg.max_per_img} respectively' \
                'Please delete max_num which will be deprecated.'
        else:
            cfg.max_per_img = cfg.max_num
    if 'nms_thr' in cfg:
        assert cfg.nms.iou_threshold == cfg.nms_thr, f'You set' \
            f' iou_threshold in nms and ' \
            f'nms_thr at the same time, but get' \
            f' {cfg.nms.iou_threshold} and {cfg.nms_thr}' \
            f' respectively. Please delete the nms_thr ' \
            f'which will be deprecated.'

    result_list = []
    # 对每一张图片的 proposal 进行 nms
    for (mlvl_proposals, mlvl_scores,
         mlvl_ids) in zip(batch_mlvl_proposals, batch_mlvl_scores,
                          batch_mlvl_ids):
        # Skip nonzero op while exporting to ONNX
        if cfg.min_bbox_size > 0 and (not torch.onnx.is_in_onnx_export()):
            w = mlvl_proposals[:, 2] - mlvl_proposals[:, 0]
            h = mlvl_proposals[:, 3] - mlvl_proposals[:, 1]
            valid_ind = torch.nonzero(
                (w >= cfg.min_bbox_size)
                & (h >= cfg.min_bbox_size),
                as_tuple=False).squeeze()
            if valid_ind.sum().item() != len(mlvl_proposals):
                mlvl_proposals = mlvl_proposals[valid_ind, :]
                mlvl_scores = mlvl_scores[valid_ind]
                mlvl_ids = mlvl_ids[valid_ind]
				# mlvl_ids 表示这个 proposal 属于第几个 FPN 层，range：[0, num_level-1]
        dets, keep = batched_nms(mlvl_proposals, mlvl_scores, mlvl_ids,
                                 cfg.nms)
        result_list.append(dets[:cfg.max_per_img])
    # 最终返回的列表是每一张图片中选出来的 proposal，每一个元素的 shape 都是 [max_per_img, 5]
    # 最后一个维度代表 proposal 的坐标和得分，这边 max_per_img 是 1000
    return result_list
```



## ROI_head



到前面为止，第一阶段的 loss 和 proposal 全部都生成了，那么现在就是要将 proposal 送入到第二阶段去了，在代码里面接下去运行的是 ROI_head 的 forward_train 函数



### ROI_head.forward_train



```python
roi_losses = self.roi_head.forward_train(x, img_metas, proposal_list,
                                            gt_bboxes, gt_labels,
                                            gt_bboxes_ignore, gt_masks,
                                            **kwargs)
```

函数传入了生成的 proposal 和 gt_bbox 的坐标以及标签，就要判断框对应的具体的类别了，下面具体来分析一下。

```python
rcnn=dict(
    assigner=dict(
        type='MaxIoUAssigner',
        pos_iou_thr=0.5,
        neg_iou_thr=0.5,
        min_pos_iou=0.5,
        match_low_quality=False,
        ignore_iof_thr=-1),
    sampler=dict(
        type='RandomSampler',
      	# 取 512 个 ROI 送去训练
        num=512,
      	# 正负样本比为 1:3
        pos_fraction=0.25,
        neg_pos_ub=-1,
        add_gt_as_proposals=True),
    pos_weight=-1,
    debug=False)
```



```python
def forward_train(self,
                    x,
                    img_metas,
                    proposal_list,
                    gt_bboxes,
                    gt_labels,
                    gt_bboxes_ignore=None,
                    gt_masks=None):
    """
    Args:
        x (list[Tensor]): list of multi-level img features.
        img_metas (list[dict]): list of image info dict where each dict
            has: 'img_shape', 'scale_factor', 'flip', and may also contain
            'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
            For details on the values of these keys see
            `mmdet/datasets/pipelines/formatting.py:Collect`.
        proposals (list[Tensors]): list of region proposals.
        gt_bboxes (list[Tensor]): Ground truth bboxes for each image with
            shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
        gt_labels (list[Tensor]): class indices corresponding to each box
        gt_bboxes_ignore (None | list[Tensor]): specify which bounding
            boxes can be ignored when computing the loss.
        gt_masks (None | Tensor) : true segmentation masks for each box
            used if the architecture supports a segmentation task.

    Returns:
        dict[str, Tensor]: a dictionary of loss components
    """
    # assign gts and sample proposals
    if self.with_bbox or self.with_mask:
        num_imgs = len(img_metas)
        if gt_bboxes_ignore is None:
            gt_bboxes_ignore = [None for _ in range(num_imgs)]
        sampling_results = []
        # 对每张图片的 gt 和 proposal 进行匹配
        for i in range(num_imgs):
            assign_result = self.bbox_assigner.assign(
                proposal_list[i], gt_bboxes[i], gt_bboxes_ignore[i],
                gt_labels[i])
            # 随机取 512 个样本拿去训练，不过为什么传入 feat 呢，也没有对应的 api？
            sampling_result = self.bbox_sampler.sample(
                assign_result,
                proposal_list[i],
                gt_bboxes[i],
                gt_labels[i],
                feats=[lvl_feat[i][None] for lvl_feat in x])
            sampling_results.append(sampling_result)

    losses = dict()
    # bbox head forward and loss
    if self.with_bbox:
        bbox_results = self._bbox_forward_train(x, sampling_results,
                                                gt_bboxes, gt_labels,
                                                img_metas)
        losses.update(bbox_results['loss_bbox'])

    # mask head forward and loss
    if self.with_mask:
        mask_results = self._mask_forward_train(x, sampling_results,
                                                bbox_results['bbox_feats'],
                                                gt_masks, img_metas)
        losses.update(mask_results['loss_mask'])

    return losses
```

 

### ROI_head._bbox_forward_train



这个函数是 ROI_head 根据第一阶段的 proposals 得出 ROI 之后进行 loss 的计算

```python
def _bbox_forward_train(self, x, sampling_results, gt_bboxes, gt_labels,
                        img_metas):
    """Run forward function and calculate loss for box head in training."""
    # 得到了 batch 中所有的 bbox 的信息，shape: (n, 5) -> [batch_ind, x1, y1, x2, y2]
    rois = bbox2roi([res.bboxes for res in sampling_results])
    # 根据特征图和 roi 进行前向传播，
    bbox_results = self._bbox_forward(x, rois)
		# 根据 sample 出来的正负样本得到这些样本对应的 gt
    bbox_targets = self.bbox_head.get_targets(sampling_results, gt_bboxes,
                                                gt_labels, self.train_cfg)
    # 计算 loss
    loss_bbox = self.bbox_head.loss(bbox_results['cls_score'],
                                    bbox_results['bbox_pred'], rois,
                                    *bbox_targets)

    bbox_results.update(loss_bbox=loss_bbox)
    return bbox_results
```



### ROI_head._bbox_forward



这个函数记录 ROI_head 中的 bbox 分支前向传播的过程

```python
def _bbox_forward(self, x, rois):
    """Box head forward function used in both training and testing."""
    # TODO: a more flexible way to decide which feature maps to use
    # 根据特征图和 proposals 的坐标取出 ROI 特征。shape：（bs*num_samples, 256, 7, 7）
    bbox_feats = self.bbox_roi_extractor(
        x[:self.bbox_roi_extractor.num_inputs], rois)
    if self.with_shared_head:
      	# 根据这些特征图前向传播得到 cls_score 和 bbox_pred
        # shape: (bs*num_samples, num_class+1), (bs*num_samples, 4*num_class)
        # 因为 reg_class_agnostic=False
        bbox_feats = self.shared_head(bbox_feats)
    		cls_score, bbox_pred = self.bbox_head(bbox_feats)

    bbox_results = dict(
        cls_score=cls_score, bbox_pred=bbox_pred, bbox_feats=bbox_feats)
    return bbox_results
```



### ROI_head.bbox_roi_extractor



下面是 Faster RCNN 中提取 ROI 特征的 config 配置，用的是 `SingleRoIExtractor` 这个类，继承自 `BaseRoIExtractor`

```python
bbox_roi_extractor=dict(
    type='SingleRoIExtractor',
    roi_layer=dict(type='RoIAlign', output_size=7, sampling_ratio=0),
    out_channels=256,
    featmap_strides=[4, 8, 16, 32])
```



注意，不是所有的特征图都会参与 ROI 特征提取，这里的话 FPN 出来之后有 5 个特征层，但是最终参与 ROI 提取的只有 4 个特征层，具体在下面代码中体现出来

```python
bbox_feats = self.bbox_roi_extractor(
  x[:self.bbox_roi_extractor.num_inputs], rois)
```

```python
self.bbox_roi_extractor.num_inputs = len(self.featmap_strides)
```

所以看到，最后一层下采样 32 倍的特征图是没有参与计算的。继续往下看 roi_extractor 的前向代码：

```python
@force_fp32(apply_to=('feats', ), out_fp16=True)
def forward(self, feats, rois, roi_scale_factor=None):
    """Forward function."""
    # self.roi_layer 是一个列表，len 为 len(feats)，这里是 4
    # out_size 是 roi 特征被提取出来之后经过 roi_pooling 或者 roi_align 统一后的尺寸（7*7）
    out_size = self.roi_layers[0].output_size
    # num_levels=4
    num_levels = len(feats)
    expand_dims = (-1, self.out_channels * out_size[0] * out_size[1])
    if torch.onnx.is_in_onnx_export():
        # Work around to export mask-rcnn to onnx
        roi_feats = rois[:, :1].clone().detach()
        roi_feats = roi_feats.expand(*expand_dims)
        roi_feats = roi_feats.reshape(-1, self.out_channels, *out_size)
        roi_feats = roi_feats * 0
    else:
      	# 创建一个全零的 placeholder，shape：(bs*num_samples, 256, 7, 7)
        roi_feats = feats[0].new_zeros(
            rois.size(0), self.out_channels, *out_size)
    # TODO: remove this when parrots supports
    if torch.__version__ == 'parrots':
        roi_feats.requires_grad = True

    if num_levels == 1:
        if len(rois) == 0:
            return roi_feats
        return self.roi_layers[0](feats[0], rois)
		# 给 batch 中的所有 proposals 都安排到一个相对应的特征图尺度上,shape:(bs*num_samples)
    target_lvls = self.map_roi_levels(rois, num_levels)

    if roi_scale_factor is not None:
        rois = self.roi_rescale(rois, roi_scale_factor)
		# 对每一个特征图尺度来说
    for i in range(num_levels):
      	# 找出被分配到这个尺度的 proposals
        mask = target_lvls == i
        if torch.onnx.is_in_onnx_export():
            # To keep all roi_align nodes exported to onnx
            # and skip nonzero op
            mask = mask.float().unsqueeze(-1).expand(*expand_dims).reshape(
                roi_feats.shape)
            roi_feats_t = self.roi_layers[i](feats[i], rois)
            roi_feats_t *= mask
            roi_feats += roi_feats_t
            continue
        # 找到被分配到这个尺度的 proposals 的索引，shape：(num_proposals_in_this_lvl)
        inds = mask.nonzero(as_tuple=False).squeeze(1)
        if inds.numel() > 0:
          	# 根据索引找到对应的 proposals 
            rois_ = rois[inds]
            # 真正根据 proposals 的坐标和 FPN 特征将 ROI 特征抠出来
            # 用的是 roi_pooling，mmcv ops 里面实现的操作
            # 由于 rois_ 里面保存了这个 proposal 在 batch 中的 index，所以可以直接
            # 去该图中对应的尺度的特征图中抠出来 ROI 特征
            # shape：(num_proposals_in_this_lvl)
            roi_feats_t = self.roi_layers[i](feats[i], rois_)
            # 把抠出来的 roi 特征图放进 placeholder 中返回
            roi_feats[inds] = roi_feats_t
        else:
            # Sometimes some pyramid levels will not be used for RoI
            # feature extraction and this will cause an incomplete
            # computation graph in one GPU, which is different from those
            # in other GPUs and will cause a hanging error.
            # Therefore, we add it to ensure each feature pyramid is
            # included in the computation graph to avoid runtime bugs.
            roi_feats += sum(
                x.view(-1)[0]
                for x in self.parameters()) * 0. + feats[i].sum() * 0.
    # shape：（bs*num_samples）
    return roi_feats

```



### ROI_head.bbox_roi_extractor.map_roi_levels



根据 proposals 的尺度对这些 proposals 重新分配 FPN 特征尺度层（openmmlab 的解释有点牵强，我的解释是：原来分配的时候是分配 5 个特征图，后面只用了 4 个特征图做 roi_pooling，少了一个特征图，所以要重新对这些 proposals 进行特征图分配）

```python
def map_roi_levels(self, rois, num_levels):
    """Map rois to corresponding feature levels by scales.

    - scale < finest_scale * 2: level 0
    - finest_scale * 2 <= scale < finest_scale * 4: level 1
    - finest_scale * 4 <= scale < finest_scale * 8: level 2
    - scale >= finest_scale * 8: level 3

    Args:
        rois (Tensor): Input RoIs, shape (k, 5).
        num_levels (int): Total level number.

    Returns:
        Tensor: Level index (0-based) of each RoI, shape (k, )
    """
    scale = torch.sqrt(
        (rois[:, 3] - rois[:, 1]) * (rois[:, 4] - rois[:, 2]))
    target_lvls = torch.floor(torch.log2(scale / self.finest_scale + 1e-6))
    target_lvls = target_lvls.clamp(min=0, max=num_levels - 1).long()
    return target_lvls
```



### ROI_head.bbox_roi_extractor.build_roi_layers



build_roi_layers 函数为每一个尺度的特征图都实例化一个 roi_pooling 对象，传入 spatial_scale 参数，是让 proposals 根据 stride 来调整大小，以此来抠出对应的特征，例如最后一层的小特征图是用来回归大的物体，那么这个 proposal 的长宽肯定很大，除以一个 stride 来将 proposal 映射到特征图上，根据映射之后的坐标抠出特征图。

```python
    def build_roi_layers(self, layer_cfg, featmap_strides):
        """Build RoI operator to extract feature from each level feature map.

        Args:
            layer_cfg (dict): Dictionary to construct and config RoI layer
                operation. Options are modules under ``mmcv/ops`` such as
                ``RoIAlign``.
            featmap_strides (List[int]): The stride of input feature map w.r.t
                to the original image size, which would be used to scale RoI
                coordinate (original image coordinate system) to feature
                coordinate system.

        Returns:
            nn.ModuleList: The RoI extractor modules for each level feature
                map.
        """

        cfg = layer_cfg.copy()
        layer_type = cfg.pop('type')
        assert hasattr(ops, layer_type)
        layer_cls = getattr(ops, layer_type)
        roi_layers = nn.ModuleList(
            [layer_cls(spatial_scale=1 / s, **cfg) for s in featmap_strides])
        return roi_layers
```



## bbox2roi



这个函数将一个 batch 的 proposals 融合在一起变成一个大列表，并且给每一个 proposal 都加上一个信息，代表这个 proposal 是从这个 batch 中的第几张图里面拿出来的。

```python
def bbox2roi(bbox_list):
    """Convert a list of bboxes to roi format.

    Args:
        bbox_list (list[Tensor]): a list of bboxes corresponding to a batch
            of images.

    Returns:
        Tensor: shape (n, 5), [batch_ind, x1, y1, x2, y2]
    """
    rois_list = []
    for img_id, bboxes in enumerate(bbox_list):
        if bboxes.size(0) > 0:
            img_inds = bboxes.new_full((bboxes.size(0), 1), img_id)
            rois = torch.cat([img_inds, bboxes[:, :4]], dim=-1)
        else:
            rois = bboxes.new_zeros((0, 5))
        rois_list.append(rois)
    # shape: （n,5），n 是这个 batch 里面所有的 proposal 之和。
    # 这里是 16*512=8192
    rois = torch.cat(rois_list, 0)
    return rois
```



## Shared2FCBBoxHead



这个是 ROI_head 中的 bbox_head 配置，因为太多了，所以另开一节写，主要就是获取训练第二阶段的 targets，将 roi_pooling 之后抠出来的特征拿去计算 loss。以下所有的 head 都在 `mmdet/models/roi_heads/bbox_heads` 里面。

```python
@HEADS.register_module()
class Shared2FCBBoxHead(ConvFCBBoxHead):

    def __init__(self, fc_out_channels=1024, *args, **kwargs):
        super(Shared2FCBBoxHead, self).__init__(
            num_shared_convs=0,
            num_shared_fcs=2,
            num_cls_convs=0,
            num_cls_fcs=0,
            num_reg_convs=0,
            num_reg_fcs=0,
            fc_out_channels=fc_out_channels,
            *args,
            **kwargs)

```

同时，Shared2FCBBoxHead 又继承自 ConvFCBBoxHead，也就是在分支上加几层卷积之后加上全连接层，而 ConvFCBBoxHead 又继承自 BBoxHead，这是 ROI_head 的基类，获取第二阶段训练的目标以及第二阶段前向传播都是通过这个类进行的。



## BBoxHead



这是最简单的 ROI_head，只有两个 fc 层用来做分类和回归，上述派生的子类可以在这个基础上实现添加几个卷积之类的改动。总之这里就是前向传播，最终分类分支  `shape: (bs*num_samples, num_class+1)`，回归分支 `shape: (bs*num_samples, 4*num_class)`, 因为 reg_class_agnostic=False

```python

@HEADS.register_module()
class BBoxHead(nn.Module):
    """Simplest RoI head, with only two fc layers for classification and
    regression respectively."""

    def __init__(self,
                 with_avg_pool=False,
                 with_cls=True,
                 with_reg=True,
                 roi_feat_size=7,
                 in_channels=256,
                 num_classes=80,
                 bbox_coder=dict(
                     type='DeltaXYWHBBoxCoder',
                     clip_border=True,
                     target_means=[0., 0., 0., 0.],
                     target_stds=[0.1, 0.1, 0.2, 0.2]),
                 reg_class_agnostic=False,
                 reg_decoded_bbox=False,
                 loss_cls=dict(
                     type='CrossEntropyLoss',
                     use_sigmoid=False,
                     loss_weight=1.0),
                 loss_bbox=dict(
                     type='SmoothL1Loss', beta=1.0, loss_weight=1.0)):
        super(BBoxHead, self).__init__()
        assert with_cls or with_reg
        self.with_avg_pool = with_avg_pool
        self.with_cls = with_cls
        self.with_reg = with_reg
        self.roi_feat_size = _pair(roi_feat_size)
        self.roi_feat_area = self.roi_feat_size[0] * self.roi_feat_size[1]
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.reg_class_agnostic = reg_class_agnostic
        self.reg_decoded_bbox = reg_decoded_bbox
        self.fp16_enabled = False

        self.bbox_coder = build_bbox_coder(bbox_coder)
        self.loss_cls = build_loss(loss_cls)
        self.loss_bbox = build_loss(loss_bbox)

        in_channels = self.in_channels
        if self.with_avg_pool:
            self.avg_pool = nn.AvgPool2d(self.roi_feat_size)
        else:
            in_channels *= self.roi_feat_area
        if self.with_cls:
            # need to add background class
            self.fc_cls = nn.Linear(in_channels, num_classes + 1)
        if self.with_reg:
            out_dim_reg = 4 if reg_class_agnostic else 4 * num_classes
            self.fc_reg = nn.Linear(in_channels, out_dim_reg)
        self.debug_imgs = None

    def init_weights(self):
        # conv layers are already initialized by ConvModule
        if self.with_cls:
            nn.init.normal_(self.fc_cls.weight, 0, 0.01)
            nn.init.constant_(self.fc_cls.bias, 0)
        if self.with_reg:
            nn.init.normal_(self.fc_reg.weight, 0, 0.001)
            nn.init.constant_(self.fc_reg.bias, 0)

    @auto_fp16()
    def forward(self, x):
        if self.with_avg_pool:
            x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        cls_score = self.fc_cls(x) if self.with_cls else None
        bbox_pred = self.fc_reg(x) if self.with_reg else None
        return cls_score, bbox_pred
```



### BBoxHead.get_targets



其实这个跟一阶段的那些 dense_head 做的事情是一样的，就是根据 sampling_results 得到的正负样本和 gt 信息来给这些样本分配需要回归的标签

```python
def get_targets(self,
                sampling_results,
                gt_bboxes,
                gt_labels,
                rcnn_train_cfg,
                # 默认是将每一张图的 target 合并在一起
                concat=True):
    """Calculate the ground truth for all samples in a batch according to
    the sampling_results.

    Almost the same as the implementation in bbox_head, we passed
    additional parameters pos_inds_list and neg_inds_list to
    `_get_target_single` function.

    Args:
        sampling_results (List[obj:SamplingResults]): Assign results of
            all images in a batch after sampling.
        gt_bboxes (list[Tensor]): Gt_bboxes of all images in a batch,
            each tensor has shape (num_gt, 4),  the last dimension 4
            represents [tl_x, tl_y, br_x, br_y].
        gt_labels (list[Tensor]): Gt_labels of all images in a batch,
            each tensor has shape (num_gt,).
        rcnn_train_cfg (obj:ConfigDict): `train_cfg` of RCNN.
        concat (bool): Whether to concatenate the results of all
            the images in a single batch.

    Returns:
        Tuple[Tensor]: Ground truth for proposals in a single image.
        Containing the following list of Tensors:

            - labels (list[Tensor],Tensor): Gt_labels for all
                proposals in a batch, each tensor in list has
                shape (num_proposals,) when `concat=False`, otherwise
                just a single tensor has shape (num_all_proposals,).
            - label_weights (list[Tensor]): Labels_weights for
                all proposals in a batch, each tensor in list has
                shape (num_proposals,) when `concat=False`, otherwise
                just a single tensor has shape (num_all_proposals,).
            - bbox_targets (list[Tensor],Tensor): Regression target
                for all proposals in a batch, each tensor in list
                has shape (num_proposals, 4) when `concat=False`,
                otherwise just a single tensor has shape
                (num_all_proposals, 4), the last dimension 4 represents
                [tl_x, tl_y, br_x, br_y].
            - bbox_weights (list[tensor],Tensor): Regression weights for
                all proposals in a batch, each tensor in list has shape
                (num_proposals, 4) when `concat=False`, otherwise just a
                single tensor has shape (num_all_proposals, 4).
    """
    # list: 每一张图片中的正样本 box
    pos_bboxes_list = [res.pos_bboxes for res in sampling_results]
    # list: 负样本 box
    neg_bboxes_list = [res.neg_bboxes for res in sampling_results]
    # list: 正样本需要回归的目标 box
    pos_gt_bboxes_list = [res.pos_gt_bboxes for res in sampling_results]
    # list: 正样本需要回归的目标类别
    pos_gt_labels_list = [res.pos_gt_labels for res in sampling_results]
    labels, label_weights, bbox_targets, bbox_weights = multi_apply(
        self._get_target_single,
        pos_bboxes_list,
        neg_bboxes_list,
        pos_gt_bboxes_list,
        pos_gt_labels_list,
        cfg=rcnn_train_cfg)

    if concat:
      	# 把 target 全部 concat 在一起，shape：（bs*num_sample, ……)
        labels = torch.cat(labels, 0)
        label_weights = torch.cat(label_weights, 0)
        bbox_targets = torch.cat(bbox_targets, 0)
        bbox_weights = torch.cat(bbox_weights, 0)
    return labels, label_weights, bbox_targets, bbox_weights
```



### BBoxHead._get_target_single



```python
def _get_target_single(self, pos_bboxes, neg_bboxes, pos_gt_bboxes,
                        pos_gt_labels, cfg):
    """Calculate the ground truth for proposals in the single image
    according to the sampling results.

    Args:
        pos_bboxes (Tensor): Contains all the positive boxes,
            has shape (num_pos, 4), the last dimension 4
            represents [tl_x, tl_y, br_x, br_y].
        neg_bboxes (Tensor): Contains all the negative boxes,
            has shape (num_neg, 4), the last dimension 4
            represents [tl_x, tl_y, br_x, br_y].
        pos_gt_bboxes (Tensor): Contains all the gt_boxes,
            has shape (num_gt, 4), the last dimension 4
            represents [tl_x, tl_y, br_x, br_y].
        pos_gt_labels (Tensor): Contains all the gt_labels,
            has shape (num_gt).
        cfg (obj:`ConfigDict`): `train_cfg` of R-CNN.

    Returns:
        Tuple[Tensor]: Ground truth for proposals
        in a single image. Containing the following Tensors:

            - labels(Tensor): Gt_labels for all proposals, has
                shape (num_proposals,).
            - label_weights(Tensor): Labels_weights for all
                proposals, has shape (num_proposals,).
            - bbox_targets(Tensor):Regression target for all
                proposals, has shape (num_proposals, 4), the
                last dimension 4 represents [tl_x, tl_y, br_x, br_y].
            - bbox_weights(Tensor):Regression weights for all
                proposals, has shape (num_proposals, 4).
    """
    num_pos = pos_bboxes.size(0)
    num_neg = neg_bboxes.size(0)
    num_samples = num_pos + num_neg

    # original implementation uses new_zeros since BG are set to be 0
    # now use empty & fill because BG cat_id = num_classes,
    # FG cat_id = [0, num_classes-1]
    # shape: (num_samples)
    labels = pos_bboxes.new_full((num_samples, ),
                                    self.num_classes,
                                    dtype=torch.long)
    label_weights = pos_bboxes.new_zeros(num_samples)
    bbox_targets = pos_bboxes.new_zeros(num_samples, 4)
    bbox_weights = pos_bboxes.new_zeros(num_samples, 4)
    if num_pos > 0:
      	# 把正样本放在前面，因为在 sampling_results 中是先 sample_pos 再 sample_neg 的
        # 这里也要按照顺序对应起来
        labels[:num_pos] = pos_gt_labels
        pos_weight = 1.0 if cfg.pos_weight <= 0 else cfg.pos_weight
        label_weights[:num_pos] = pos_weight
        if not self.reg_decoded_bbox:
            pos_bbox_targets = self.bbox_coder.encode(
                pos_bboxes, pos_gt_bboxes)
        else:
            # When the regression loss (e.g. `IouLoss`, `GIouLoss`)
            # is applied directly on the decoded bounding boxes, both
            # the predicted boxes and regression targets should be with
            # absolute coordinate format.
            pos_bbox_targets = pos_gt_bboxes
        # 同样也是把正样本放在前面
        bbox_targets[:num_pos, :] = pos_bbox_targets
        bbox_weights[:num_pos, :] = 1
    if num_neg > 0:
        label_weights[-num_neg:] = 1.0
		# 最终，label_weights 里面全是 1， 因为正负样本都要参与计算分类 loss
    # bbox_weights 中正样本才为 1，负样本为 0
    # bbox_targets 可能正样本的 target 也全是 0，因为可能前期样本不够，gt 拿来当作 proposal 了
    # 导致 proposal 到 gt 之间的偏移为 0
    return labels, label_weights, bbox_targets, bbox_weights
```



### BBoxHead.loss



这里就是对第二阶段 head 部分做分类和回归的 loss，大同小异，不过下面 reg_class_agnostic 那里的写法可以学一下。

```python
@force_fp32(apply_to=('cls_score', 'bbox_pred'))
def loss(self,
            cls_score,# (bs*num_samples, num_class+1)
            bbox_pred,# (bs*num_samples, num_class*4)
            rois,# (bs*num_samples, 5)
            labels,# (bs*num_samples)
            label_weights,
            bbox_targets,# (bs*num_samples, 4)
            bbox_weights,
            reduction_override=None):
    losses = dict()
    if cls_score is not None:
        avg_factor = max(torch.sum(label_weights > 0).float().item(), 1.)
        if cls_score.numel() > 0:
            losses['loss_cls'] = self.loss_cls(
                cls_score,
                labels,
                label_weights,
                avg_factor=avg_factor,
                reduction_override=reduction_override)
            losses['acc'] = accuracy(cls_score, labels)
    if bbox_pred is not None:
        bg_class_ind = self.num_classes
        # 0~self.num_classes-1 are FG, self.num_classes is BG
        # 找出正样本所在的索引，shape:(num_pos)
        pos_inds = (labels >= 0) & (labels < bg_class_ind)
        # do not perform bounding box regression for BG anymore.
        if pos_inds.any():
            if self.reg_decoded_bbox:
                # When the regression loss (e.g. `IouLoss`,
                # `GIouLoss`, `DIouLoss`) is applied directly on
                # the decoded bounding boxes, it decodes the
                # already encoded coordinates to absolute format.
                # 这里的 rois 其实就代表了 anchor，因为这个 roi 也就是从第一阶段的 anchor 中产生的
                bbox_pred = self.bbox_coder.decode(rois[:, 1:], bbox_pred)
            # reg_class_agnostic 这里是 False
            if self.reg_class_agnostic:
                pos_bbox_pred = bbox_pred.view(
                    bbox_pred.size(0), 4)[pos_inds.type(torch.bool)]
            else:
              	# bbox_pred：（bs*num_samples，num_class*4）
                # 这个写法好啊，取出正样本中该样本所属 gt 类别的预测框信息
                # shape：(num_pos, 4)
                pos_bbox_pred = bbox_pred.view(
                    bbox_pred.size(0), -1,
                    4)[pos_inds.type(torch.bool),
                        labels[pos_inds.type(torch.bool)]]
            losses['loss_bbox'] = self.loss_bbox(
                pos_bbox_pred,
                bbox_targets[pos_inds.type(torch.bool)],
                bbox_weights[pos_inds.type(torch.bool)],
                avg_factor=bbox_targets.size(0),
                reduction_override=reduction_override)
        else:
            losses['loss_bbox'] = bbox_pred[pos_inds].sum()
    return losses
```



## reference



https://zhuanlan.zhihu.com/p/349807581
