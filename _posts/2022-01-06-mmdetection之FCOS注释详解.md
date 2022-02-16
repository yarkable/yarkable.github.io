---
layout: post
title: mmdetection之FCOS注释详解
subtitle: 
date: 2022-01-06
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



本文记录 mmdetection 对 FCOS 训练的流程，包括标签获取，anchor 生成，前向训练，以及各步骤中 tensor 的形状，仅供复习用处。mmdetection 版本为 2.11.0。



## 整体流程



1. mmdet/models/detectors/base.py:forward_train()
2. mmdet/models/detectors/single_stage.py:extract_feat()
3. mmdet/models/dense_heads/fcos_head.py:forward_train()
4. fcos_head.py 没有重写 forward_train()，所以调用的是 mmdet/models/dense_heads/base_dense_head.py:forward_train()
5. 4 中的 forward_train() 调用 `self(x)` ，用到了 mmdet/models/dense_heads/fcos_head.py:forward()
6. 第 5 步前向结束得到 head 预测的输出，在 mmdet/models/dense_heads/fcos_head.py:loss() 中计算 loss

## forward



首先，这里重写了 dense_head 的 forward 函数，因为预测值相比之前多了一个 centerness，并且由于一些 trick，在前向的时候选择性加上 trick。与 RetinaNet 一样，FCOS 检测器 head 在各个特征图上是共享的，但是由于不同特征图对应的回归值范围差异较大，可能学习成本大，所以最早版本 FCOS 在 regression 分支最后输出是乘以一个可学习的 scale 值以解决这个问题。但是现在的版本中，前面已经提到每个特征图上的回归值其实是已经除以特征图的 stride 进行缩放，这样就和学习一个 scale 值基本等同，所以加不加这个策略都可以。

```python
def forward(self, feats):
    """Forward features from the upstream network.

    Args:
        feats (tuple[Tensor]): Features from the upstream network, each is
            a 4D-tensor.

    Returns:
        tuple:
            cls_scores (list[Tensor]): Box scores for each scale level, \
                each is a 4D-tensor, the channel number is \
                num_points * num_classes.
            bbox_preds (list[Tensor]): Box energies / deltas for each \
                scale level, each is a 4D-tensor, the channel number is \
                num_points * 4.
            centernesses (list[Tensor]): centerness for each scale level, \
                each is a 4D-tensor, the channel number is num_points * 1.
    """
    return multi_apply(self.forward_single, feats, self.scales,
                       self.strides)

def forward_single(self, x, scale, stride):
    """Forward features of a single scale level.

    Args:
        x (Tensor): FPN feature maps of the specified stride.
        scale (:obj: `mmcv.cnn.Scale`): Learnable scale module to resize
            the bbox prediction.
        stride (int): The corresponding stride for feature maps, only
            used to normalize the bbox prediction when self.norm_on_bbox
            is True.

    Returns:
        tuple: scores for each class, bbox predictions and centerness \
            predictions of input feature maps.
    """
    cls_score, bbox_pred, cls_feat, reg_feat = super().forward_single(x)
    if self.centerness_on_reg:
        centerness = self.conv_centerness(reg_feat)
    else:
        centerness = self.conv_centerness(cls_feat)
    # scale the bbox_pred of different level
    # float to avoid overflow when enabling FP16
    # 这里 scale 是一个 nn.Parameter(),初始化为 1，每一个特征图尺度都有一个 scale
    # 就是防止网络每一层需要回归的范围变化太大，而检测头的卷积是共享的，因此让网络自己学习一个参数，更好地回归
    bbox_pred = scale(bbox_pred).float()
    # norm_on_bbox 就是说正样本回归的目标距离需要除以当前的 stride，让每一层的回归范围差不多
    # 相当于起到了 scale 的作用
    if self.norm_on_bbox:
        # 用 ReLU 保证了输出非负
        bbox_pred = F.relu(bbox_pred)
        # 推理的时候就把预测的距离乘以 stride 进行解码
        if not self.training:
            bbox_pred *= stride
    else:
        # 让预测的回归距离不会小于等于 0
        # 训练和测试都会调用这个函数，所以训练和测试保持一致，都对 bbox_pred 进行了指数操作
        # 论文中有这个公式，exp(Si*x)
        bbox_pred = bbox_pred.exp()
    return cls_score, bbox_pred, centerness
```



## loss 函数



loss 函数和 PAA 一样，没有经过 loss_single 进行汇聚，直接通过单个 loss 函数计算整个 batch 的损失。这个和 anchor_based 的方法有点不一样的地方就是这里不需要计算越界的 anchor，因为 anchor point 肯定是生成在图像里面的，所以这里就可以看到没有计算 valid anchor，计算 loss 时也不需要生成 label_weight_list 和 box_weight_list。



```python
@force_fp32(apply_to=('cls_scores', 'bbox_preds', 'centernesses'))
def loss(self,
         cls_scores,
         bbox_preds,
         centernesses,
         gt_bboxes,
         gt_labels,
         img_metas,
         gt_bboxes_ignore=None):
    """Compute loss of the head.

    Args:
        cls_scores (list[Tensor]): Box scores for each scale level,
            each is a 4D-tensor, the channel number is
            num_points * num_classes.
        bbox_preds (list[Tensor]): Box energies / deltas for each scale
            level, each is a 4D-tensor, the channel number is
            num_points * 4.
        centernesses (list[Tensor]): centerness for each scale level, each
            is a 4D-tensor, the channel number is num_points * 1.
        gt_bboxes (list[Tensor]): Ground truth bboxes for each image with
            shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
        gt_labels (list[Tensor]): class indices corresponding to each box
        img_metas (list[dict]): Meta information of each image, e.g.,
            image size, scaling factor, etc.
        gt_bboxes_ignore (None | list[Tensor]): specify which bounding
            boxes can be ignored when computing the loss.

    Returns:
        dict[str, Tensor]: A dictionary of loss components.
    """
    assert len(cls_scores) == len(bbox_preds) == len(centernesses)
    featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]
    # 首先根据 get_points 生成所有图片中的 point，就像 anchor_base 的方法要先生成 anchor 一样
    # 得到的是一个列表，每一个元素是这个特征图尺度的所有 anchor point 的坐标，跟 batch_szie 没有关系
    # [torch.Size([13600, 2]), torch.Size([3400, 2]), torch.Size([850, 2]), torch.Size([221, 2]), torch.Size([63, 2])] 
    all_level_points = self.get_points(featmap_sizes, bbox_preds[0].dtype,
                                       bbox_preds[0].device)
    
    # 得到了每一张图片中所有 anchor point 分配的标签以及回归的距离（真实距离，不经过编码）
    # labels 是个 len = 5 的列表，每一个元素的 shape 为 ( batch_size * num_points_in_that_level) 
    labels, bbox_targets = self.get_targets(all_level_points, gt_bboxes,
                                            gt_labels)
	
    # 这么多张图片
    num_imgs = cls_scores[0].size(0)
    # flatten cls_scores, bbox_preds and centerness
    # 一个列表，每个元素代表当前 FPN 层的所有 anchor 的信息 shape: (BHW, C)，下面以此类推
    flatten_cls_scores = [
        cls_score.permute(0, 2, 3, 1).reshape(-1, self.cls_out_channels)
        for cls_score in cls_scores
    ]
    flatten_bbox_preds = [
        bbox_pred.permute(0, 2, 3, 1).reshape(-1, 4)
        for bbox_pred in bbox_preds
    ]
    flatten_centerness = [
        centerness.permute(0, 2, 3, 1).reshape(-1)
        for centerness in centernesses
    ]
    # 将一个 batch 的所有 anchor 聚集在一起，shape: (batch_all_points, C)
    flatten_cls_scores = torch.cat(flatten_cls_scores)
    # shape: (batch_all_points, 4)
    flatten_bbox_preds = torch.cat(flatten_bbox_preds)
    # shape: (batch_all_points)
    flatten_centerness = torch.cat(flatten_centerness)
    # shape: (batch_all_points)
    flatten_labels = torch.cat(labels)
    # shape: (batch_all_points, 4)
    flatten_bbox_targets = torch.cat(bbox_targets)
    # repeat points to align with bbox_preds
    # shape: (batch_all_points, 2)
    flatten_points = torch.cat(
        [points.repeat(num_imgs, 1) for points in all_level_points])

    # FG cat_id: [0, num_classes -1], BG cat_id: num_classes
    bg_class_ind = self.num_classes
    # 找到正样本 anchor point 对应的索引, shape: (num_pos_anchors)
    pos_inds = ((flatten_labels >= 0)
                & (flatten_labels < bg_class_ind)).nonzero().reshape(-1)
    # 一个 batch 中正样本的数量
    num_pos = torch.tensor(
        len(pos_inds), dtype=torch.float, device=bbox_preds[0].device)
    num_pos = max(reduce_mean(num_pos), 1.0)
    # 正负样本都参与计算分类 loss
    loss_cls = self.loss_cls(
        flatten_cls_scores, flatten_labels, avg_factor=num_pos)
	
    # 得到正样本 anchor point 的预测值
    pos_bbox_preds = flatten_bbox_preds[pos_inds]
    pos_centerness = flatten_centerness[pos_inds]

    if len(pos_inds) > 0:
        # 得到正样本 anchor point 匹配的目标
        pos_bbox_targets = flatten_bbox_targets[pos_inds]
        # 计算出正样本需要回归的 centerness 目标
        pos_centerness_targets = self.centerness_target(pos_bbox_targets)
        pos_points = flatten_points[pos_inds]
        # 根据正样本 anchor point 所在的位置以及预测出来的四条边来将折四条边还原成一个 format: (x1, y1, x2, y2) 的 box
        pos_decoded_bbox_preds = distance2bbox(pos_points, pos_bbox_preds)
        # 同理，还原出正样本应回归的目标 box
        pos_decoded_target_preds = distance2bbox(pos_points,
                                                 pos_bbox_targets)
        # centerness weighted iou loss
        centerness_denorm = max(
            reduce_mean(pos_centerness_targets.sum().detach()), 1e-6)
        # 只有正样本参与计算回归 loss
        loss_bbox = self.loss_bbox(
            pos_decoded_bbox_preds,
            pos_decoded_target_preds,
            weight=pos_centerness_targets,
            avg_factor=centerness_denorm)
        # 同样只有正样本参与计算 centerness 的损失
        loss_centerness = self.loss_centerness(
            pos_centerness, pos_centerness_targets, avg_factor=num_pos)
    else:
        loss_bbox = pos_bbox_preds.sum()
        loss_centerness = pos_centerness.sum()

    return dict(
        loss_cls=loss_cls,
        loss_bbox=loss_bbox,
        loss_centerness=loss_centerness)
```



## get_points



这个调用的是 `anchor_free_head.py` 的方法，是根据特征图级别对每一个特征图都生成 anchor_point

```python
def get_points(self, featmap_sizes, dtype, device, flatten=False):
    """Get points according to feature map sizes.

    Args:
        featmap_sizes (list[tuple]): Multi-level feature map sizes.
        dtype (torch.dtype): Type of points.
        device (torch.device): Device of points.

    Returns:
        tuple: points of each image.
    """
    mlvl_points = []
    #  对每个特征图计算 anchor point
    for i in range(len(featmap_sizes)):
        # 返回一个列表，装着每一层的 anchor point
        mlvl_points.append(
            self._get_points_single(featmap_sizes[i], self.strides[i],
                                    dtype, device, flatten))
    return mlvl_points
```



## _get_points_single



这个函数首先调用 anchor_free_head 里面的 `_get_points_single` 得到一个特征图上的所有点，然后再根据 stride 将点来映射到原图上的坐标以判断这个点是不是在 gt_bbox 里面（这里是相对于原图的坐标）

```python
strides=[8, 16, 32, 64, 128]
```

```python
def _get_points_single(self,
                       featmap_size,
                       stride,
                       dtype,
                       device,
                       flatten=False):
    """Get points according to feature map sizes."""
	# (100, 152)， (100, 152)
    y, x = super()._get_points_single(featmap_size, stride, dtype, device)
    # 然后再将点的 x y 坐标拼接在一起形成点的坐标，shape: (15200, 2)
    # 注意这里的 " + stride // 2 " 是不能省去的，加上了之后的 anchor 点才是中心的位置，否则都是从最左上角开始的
    # 更新! 其实不加上偏移也行，因为网络学习的是该点处的 anchor box 到 gt 的差距，这 0.5 个像素的差距造成的差异让网络来学习的话很简单
    # 其实有点像 anchor-based 的方法计算预测的中心点到 gt 的中心点的 delta，理解了之后就知道加不加偏移其实没啥太大关系，AutoAssign 就没有加
    points = torch.stack((x.reshape(-1) * stride, y.reshape(-1) * stride),
                         dim=-1) + stride // 2
    return points
```



 anchor_free_head 里面的 `_get_points_single` 根据特征图上点的位置，对每一个点生成对应的坐标（这里是相对于原图的坐标）

```python
def _get_points_single(self,
                       featmap_size,
                       stride,
                       dtype,
                       device,
                       flatten=False):
    """Get points of a single scale level."""
    # (100，152)
    h, w = featmap_size
    # (100, 152)
    x_range = torch.arange(w, dtype=dtype, device=device)
    y_range = torch.arange(h, dtype=dtype, device=device)
    # (100, 152)
    # x: 0-151, y: 0-99
    y, x = torch.meshgrid(y_range, x_range)
    if flatten:
        y = y.flatten()
        x = x.flatten()
    return y, x
```



## get_targets



这个函数返回每一个 FPN 层的 anchor point 要回归的目标以及他们的类别标签，注意这个函数的参数中 points 是基于特征图级别的，gt_bboxes_list 和 gt_labels_list 都是基于图像级别的

```python
def get_targets(self, points, gt_bboxes_list, gt_labels_list):
    """Compute regression, classification and centerness targets for points
    in multiple images.

    Args:
        points (list[Tensor]): Points of each fpn level, each has shape
            (num_points, 2).
        gt_bboxes_list (list[Tensor]): Ground truth bboxes of each image,
            each has shape (num_gt, 4).
        gt_labels_list (list[Tensor]): Ground truth labels of each box,
            each has shape (num_gt,).

    Returns:
        tuple:
            concat_lvl_labels (list[Tensor]): Labels of each level. \
            concat_lvl_bbox_targets (list[Tensor]): BBox targets of each \
                level.
    """
    assert len(points) == len(self.regress_ranges)
    num_levels = len(points)
    # expand regress ranges to align with points
    # self.regress_range: ((-1, 64), (64, 128), (128, 256), (256, 512), (512, 100000000.0))
    # 这里的作用其实就是生成每一层的 anchor point 所能回归的最大和最小范围，看不懂的我就来一步一步 debug
    # expanded_regress_ranges[0].unique(): (-1, 64), expanded_regress_ranges[0].shape: torch.Size([13600, 2]) 
    expanded_regress_ranges = [
        points[i].new_tensor(self.regress_ranges[i])[None].expand_as(
            points[i]) for i in range(num_levels)
    ]
    # concat all levels points and regress ranges
    # 将这张图片上的所有 anchor point 和他们应该回归的范围 concat 起来
    concat_regress_ranges = torch.cat(expanded_regress_ranges, dim=0)
    concat_points = torch.cat(points, dim=0)

    # the number of points per img, per lvl
    # 列表，每一特征图层面上的 anchor point 的个数
    num_points = [center.size(0) for center in points]

    # get labels and bbox_targets of each image
    # 得到了每一张图的 anchor point 被分配的标签和需要回归的四条边的距离
    labels_list, bbox_targets_list = multi_apply(
        self._get_target_single,
        gt_bboxes_list,
        gt_labels_list,
        points=concat_points,
        regress_ranges=concat_regress_ranges,
        num_points_per_lvl=num_points)

    # split to per img, per level
    # labels_list 此时变成了一个 len 为 num_img 的列表，列表中的每一个元素都是一个 len 为 5 的tuple
    # tuple 里的每一个元素都是该张图片在该特征图阶段的所有 anchor 数目
    labels_list = [labels.split(num_points, 0) for labels in labels_list]
    # 同理，bbox_targets_list 类型是 List[Tuple[Tensor]]，每个 tensor 的 shape 为 (num_points_in_that_level, 4)
    bbox_targets_list = [
        bbox_targets.split(num_points, 0)
        for bbox_targets in bbox_targets_list
    ]

    # concat per level image
    # 将标签以特征图级别进行 concat，得到整个 batch 中每一特征图阶段的所有 anchor point
    # concat_lvl_labels 是个 len = 5 的列表，每一个元素的 shape 为 ( batch_size * num_points_in_that_level) 
    concat_lvl_labels = []
    concat_lvl_bbox_targets = []
    for i in range(num_levels):
        concat_lvl_labels.append(
            torch.cat([labels[i] for labels in labels_list]))
        bbox_targets = torch.cat(
            [bbox_targets[i] for bbox_targets in bbox_targets_list])
        if self.norm_on_bbox:
            bbox_targets = bbox_targets / self.strides[i]
        concat_lvl_bbox_targets.append(bbox_targets)
    return concat_lvl_labels, concat_lvl_bbox_targets
```



## _get_target_single





```python
def _get_target_single(self, gt_bboxes, gt_labels, points, regress_ranges,
                       num_points_per_lvl):
    """Compute regression and classification targets for a single image."""
    num_points = points.size(0)
    num_gts = gt_labels.size(0)
    if num_gts == 0:
        return gt_labels.new_full((num_points,), self.num_classes), \
               gt_bboxes.new_zeros((num_points, 4))
	
    # 计算出该图片中每一个 gt_box 的面积，shape：num_gt_box
    # 方便 anchor 落在多个 gt_box 中最终决策为回归面积最小的 gt_box
    areas = (gt_bboxes[:, 2] - gt_bboxes[:, 0]) * (
        gt_bboxes[:, 3] - gt_bboxes[:, 1])
    # TODO: figure out why these two are different
    # areas = areas[None].expand(num_points, num_gts)
    # 扩展维度，shape: num_gt_box -> (1, num_gt_box) -> (num_anchor_point, num_gt)
    # 之所以这样扩展是因为后面每一个 anchor 都要和每一个 gt_box 进行计算，看看匹配哪一个 gt_box，这样子的话方便很多
    areas = areas[None].repeat(num_points, 1)
    # 扩展维度，shape: (num_anchor_point, 2) -> (num_anchor_point, 1, 2) -> (num_anchor_point, num_gt, 2)
    regress_ranges = regress_ranges[:, None, :].expand(
        num_points, num_gts, 2)
    # 扩展维度，shape: (num_gt, 4) -> (1, num_gt, 4) -> (num_anchor_point, num_gt, 4) 
    gt_bboxes = gt_bboxes[None].expand(num_points, num_gts, 4)
    xs, ys = points[:, 0], points[:, 1]
    # 扩展维度，shape: num_anchor_point -> (num_anchor_point, 1) -> (num_anchor_point, num_gt) 
    xs = xs[:, None].expand(num_points, num_gts)
    ys = ys[:, None].expand(num_points, num_gts)

    # 分别是所有 anchor point 到 每一个 gt_box 的四条边的距离，shape: (num_anchor_point, num_gt) 
    left = xs - gt_bboxes[..., 0]
    right = gt_bboxes[..., 2] - xs
    top = ys - gt_bboxes[..., 1]
    bottom = gt_bboxes[..., 3] - ys
    # 将距离聚成一条向量，shape: (num_anchor_point, num_gt, 4) 
    bbox_targets = torch.stack((left, top, right, bottom), -1)
	
    # 只有当 anchor point 在 gt_box 的中间一小部分的时候才算作正样本，这样的话减少了很多存在于 gt 背景区域的样本
    if self.center_sampling:
        # condition1: inside a `center bbox`
        radius = self.center_sample_radius
        # 得到每一个 gt_bbox 的中心点
        center_xs = (gt_bboxes[..., 0] + gt_bboxes[..., 2]) / 2
        center_ys = (gt_bboxes[..., 1] + gt_bboxes[..., 3]) / 2
        # 占位，shape: (num_anchor_point, num_gt, 4)
        center_gts = torch.zeros_like(gt_bboxes)
        # 保存每一个 anchor point 当前的 stride，shape: (num_anchor_point, num_gt)
        stride = center_xs.new_zeros(center_xs.shape)

        # project the points on current lvl back to the `original` sizes
        # 只将 gt_bbox 周围 stride*radius 范围内的点当成正样本
        lvl_begin = 0
        for lvl_idx, num_points_lvl in enumerate(num_points_per_lvl):
            lvl_end = lvl_begin + num_points_lvl
            stride[lvl_begin:lvl_end] = self.strides[lvl_idx] * radius
            lvl_begin = lvl_end

        x_mins = center_xs - stride
        y_mins = center_ys - stride
        x_maxs = center_xs + stride
        y_maxs = center_ys + stride
        # 将原来的 gt_box 根据新的范围给缩小
        center_gts[..., 0] = torch.where(x_mins > gt_bboxes[..., 0],
                                         x_mins, gt_bboxes[..., 0])
        center_gts[..., 1] = torch.where(y_mins > gt_bboxes[..., 1],
                                         y_mins, gt_bboxes[..., 1])
        center_gts[..., 2] = torch.where(x_maxs > gt_bboxes[..., 2],
                                         gt_bboxes[..., 2], x_maxs)
        center_gts[..., 3] = torch.where(y_maxs > gt_bboxes[..., 3],
                                         gt_bboxes[..., 3], y_maxs)
		
        # 得到 anchor point 到新的 gt_box 的四条边的距离
        cb_dist_left = xs - center_gts[..., 0]
        cb_dist_right = center_gts[..., 2] - xs
        cb_dist_top = ys - center_gts[..., 1]
        cb_dist_bottom = center_gts[..., 3] - ys
        center_bbox = torch.stack(
            (cb_dist_left, cb_dist_top, cb_dist_right, cb_dist_bottom), -1)
        inside_gt_bbox_mask = center_bbox.min(-1)[0] > 0
    else:
        # condition1: inside a gt bbox
        # 找出在 gt_box 内部的 anchor 的 mask，shape: (num_anchor_point, num_gt) 
        inside_gt_bbox_mask = bbox_targets.min(-1)[0] > 0

    # condition2: limit the regression range for each location
    # 最长的回归边要在该 anchor 能回归的范围之内， inside_regress_range: (num_anchor_point, num_gt) 
    max_regress_distance = bbox_targets.max(-1)[0]
    inside_regress_range = (
        (max_regress_distance >= regress_ranges[..., 0])
        & (max_regress_distance <= regress_ranges[..., 1]))

    # if there are still more than one objects for a location,
    # we choose the one with minimal area
    # 将不与这个 anchor point 匹配的 gt_box 的 area 置为无穷大，方便下面如果一个 anchor 和多个 gt_box 都能匹配的话选择一个面积最小的 gt_box
    areas[inside_gt_bbox_mask == 0] = INF
    areas[inside_regress_range == 0] = INF
    # 找到每一个 anchor 匹配的 gt_box (面积最小的那个)，shape: num_anchor_point
    #  min_area_inds.unique() -> 第几个 gt_box ->  tensor([0, 1, 2, 3, 4], device='cuda:0')
    min_area, min_area_inds = areas.min(dim=1)
	
    # 给每一个正样本都赋值与回归对象相应的类别标签
    labels = gt_labels[min_area_inds]
    # 负样本的标签为背景
    labels[min_area == INF] = self.num_classes  # set as BG
    # 找到每一个 anchor 需要回归的距离
    # [range(num_points), min_area_inds] 这个写法很妙，将 (num_anchor_point, num_gt, 4)  -> (num_anchor_point, 4)
    # 但是这个距离好像没有分正负样本？没错！
    bbox_targets = bbox_targets[range(num_points), min_area_inds]

    return labels, bbox_targets
```



## centerness_target



计算出正样本 anchor point 需要回归的 centerness 目标

```python
def centerness_target(self, pos_bbox_targets):
    """Compute centerness targets.

    Args:
        pos_bbox_targets (Tensor): BBox targets of positive bboxes in shape
            (num_pos, 4)

    Returns:
        Tensor: Centerness target.
    """
    # only calculate pos centerness targets, otherwise there may be nan
    left_right = pos_bbox_targets[:, [0, 2]]
    top_bottom = pos_bbox_targets[:, [1, 3]]
    centerness_targets = (
        left_right.min(dim=-1)[0] / left_right.max(dim=-1)[0]) * (
            top_bottom.min(dim=-1)[0] / top_bottom.max(dim=-1)[0])
    return torch.sqrt(centerness_targets)
```



## distance2bbox



就是根据 anchor point 所在的位置以及预测出来的四条边来将折四条边还原成一个 format: (x1, y1, x2, y2) 的 box

```python
def distance2bbox(points, distance, max_shape=None):
    """Decode distance prediction to bounding box.

    Args:
        points (Tensor): Shape (B, N, 2) or (N, 2).
        distance (Tensor): Distance from the given point to 4
            boundaries (left, top, right, bottom). Shape (B, N, 4) or (N, 4)
        max_shape (Sequence[int] or torch.Tensor or Sequence[
            Sequence[int]],optional): Maximum bounds for boxes, specifies
            (H, W, C) or (H, W). If priors shape is (B, N, 4), then
            the max_shape should be a Sequence[Sequence[int]]
            and the length of max_shape should also be B.

    Returns:
        Tensor: Boxes with shape (N, 4) or (B, N, 4)
    """
    x1 = points[..., 0] - distance[..., 0]
    y1 = points[..., 1] - distance[..., 1]
    x2 = points[..., 0] + distance[..., 2]
    y2 = points[..., 1] + distance[..., 3]

    bboxes = torch.stack([x1, y1, x2, y2], -1)

    if max_shape is not None:
        if not isinstance(max_shape, torch.Tensor):
            max_shape = x1.new_tensor(max_shape)
        max_shape = max_shape[..., :2].type_as(x1)
        if max_shape.ndim == 2:
            assert bboxes.ndim == 3
            assert max_shape.size(0) == bboxes.size(0)

        min_xy = x1.new_tensor(0)
        max_xy = torch.cat([max_shape, max_shape],
                           dim=-1).flip(-1).unsqueeze(-2)
        bboxes = torch.where(bboxes < min_xy, min_xy, bboxes)
        bboxes = torch.where(bboxes > max_xy, max_xy, bboxes)

    return bboxes
```







## get_bboxes





## _get_bboxes







## reference



[FCOS算法的原理与实现 - 知乎 (zhihu.com)](https://zhuanlan.zhihu.com/p/156112318)
