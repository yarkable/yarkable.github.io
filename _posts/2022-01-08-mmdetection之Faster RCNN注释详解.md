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



1. 





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

    # Faster 是有 RPN 的，Fast RCNN 才没有
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



所以就和普通的 anchor_based 的算法一样，首先在每个特征图上都生成一堆 anchor，然后将所有 anchor 和所有 gt 一一匹配，互相找到 IoU 最大的索引，然后确定回归和分类目标，只不过分类目标是 0 或者 1，0代表正样本，1 代表负样本，在 anchor_head 中有写道

```python
if gt_labels is None:
    # Only rpn gives gt_labels as None
    # Foreground is the first class since v2.5.0
    labels[pos_inds] = 0
```



### RPNHead.train_cfg.assigner



TODO







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





### RPNHead.train_cfg.assigner





### RPNHead.get_bboxes





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
    for idx in range(len(cls_scores)):
        rpn_cls_score = cls_scores[idx]
        rpn_bbox_pred = bbox_preds[idx]
        assert rpn_cls_score.size()[-2:] == rpn_bbox_pred.size()[-2:]
        rpn_cls_score = rpn_cls_score.permute(0, 2, 3, 1)
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

            elif scores.shape[-1] > cfg.nms_pre:
                ranked_scores, rank_inds = scores.sort(descending=True)
                topk_inds = rank_inds[:, :cfg.nms_pre]
                scores = ranked_scores[:, :cfg.nms_pre]
                batch_inds = torch.arange(batch_size).view(
                    -1, 1).expand_as(topk_inds)
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

        dets, keep = batched_nms(mlvl_proposals, mlvl_scores, mlvl_ids,
                                 cfg.nms)
        result_list.append(dets[:cfg.max_per_img])
    return result_list
```







## loss 函数





 

