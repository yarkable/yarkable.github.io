---
layout: post
title: mmdetection之RetinaNet注释详解
subtitle: 
date: 2022-01-02
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



本文记录 mmdetection 对 RetinaNet 训练的流程，包括标签获取，anchor 生成，前向训练，以及各步骤中 tensor 的形状，仅供复习用处。mmdetection 版本为 2.11.0





## loss 函数



loss 函数传入的参数有网络每一层的预测（shape: B,AC,H,W），以及每一张图片中的 gt 框（format: x1,y1,x2,y2）和所属类别，img_metas 也是一个 list，每一个元素代表了一张图中的一些信息。

```python
ipdb> len(gt_bboxes)
16
ipdb> gt_bboxes[1].shape
torch.Size([17, 4])
ipdb> gt_bboxes[1][1]
tensor([400.4000, 884.4429, 524.7333, 961.1837], device='cuda:0')
ipdb> gt_labels[1]
tensor([74, 13, 13, 13, 13, 13, 13,  0,  0,  0,  0, 13,  0,  0,  0, 13, 13],
       device='cuda:0')
ipdb> img_metas[3].keys()
dict_keys(['filename', 'ori_filename', 'ori_shape', 'img_shape', 'pad_shape', 'scale_factor', 'flip', 'flip_direction', 'img_norm_cfg', 'batch_input_shape'])
ipdb> img_metas[3]['ori_shape']
(612, 612, 3)
ipdb> img_metas[3]['img_shape']
(800, 800, 3)
ipdb> img_metas[3]['scale_factor_shape']
*** KeyError: 'scale_factor_shape'
ipdb> img_metas[3]['scale_factor']
array([1.3071896, 1.3071896, 1.3071896, 1.3071896], dtype=float32)
ipdb> img_metas[3]['pad_shape']
(800, 800, 3)
ipdb> img_metas[3]['batch_input_shape']
(1216, 800)

```



```python
@force_fp32(apply_to=('cls_scores', 'bbox_preds'))
def loss(self,
         cls_scores,
         bbox_preds,
         gt_bboxes,
         gt_labels,
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
            gt_labels (list[Tensor]): class indices corresponding to each box
            img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            gt_bboxes_ignore (None | list[Tensor]): specify which bounding
                boxes can be ignored when computing the loss. Default: None

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """

    # 每一层特征图的尺寸 H*W
    # [torch.Size([152, 100]), torch.Size([76, 50]), torch.Size([38, 25]), torch.Size([19, 13]), torch.Size([10, 7])]
    featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]
    assert len(featmap_sizes) == self.anchor_generator.num_levels

    device = cls_scores[0].device
    # 返回两个 List[List[Tensor]]最外面 list 的 size 为 batch_size，里面的是 FPN 特征图的个数，
    # 最里面的 tensor 就是每一个特征图中所含有的 anchor 数目，shape 为 (A, 4)
    anchor_list, valid_flag_list = self.get_anchors(
        featmap_sizes, img_metas, device=device)
    label_channels = self.cls_out_channels if self.use_sigmoid_cls else 1
    # get_targets 获取到所有 anchor 被匹配到的 gt_box 的偏移以及对应的类别信息
    cls_reg_targets = self.get_targets(
        anchor_list,
        valid_flag_list,
        gt_bboxes,
        img_metas,
        gt_bboxes_ignore_list=gt_bboxes_ignore,
        gt_labels_list=gt_labels,
        label_channels=label_channels)
    if cls_reg_targets is None:
        return None
    (labels_list, label_weights_list, bbox_targets_list, bbox_weights_list,
     num_total_pos, num_total_neg) = cls_reg_targets
	
    # sampling 指的是对所有样本进行筛选，防止负样本太多，如果分类损失函数是 'FocalLoss', 'GHMC', 'QualityFocalLoss'
    # 则不用 sampling
    num_total_samples = (
        num_total_pos + num_total_neg if self.sampling else num_total_pos)

    # anchor number of multi levels
    # 每一层特征图中 anchor 的数目
    # [136800, 34200, 8550, 2223, 630]
    num_level_anchors = [anchors.size(0) for anchors in anchor_list[0]]
    # concat all level anchors and flags to a single tensor
    concat_anchor_list = []
    # 将每一张图的所有 anchor 都 concat 在一起
    for i in range(len(anchor_list)):
        concat_anchor_list.append(torch.cat(anchor_list[i]))
    # concat_anchor_list 此时是 [torch.Size([182403, 4])]*batch_size 的一个列表
    # image_to_levels 将一张图上所有 anchor 转化成每一个特征图上的 anchor
    # all_anchor_list 是一个列表，里面每一个元素就是 batch 中一个特征图上的所有 anchor，
    # shape 为 (B,All_anchors, 4)
    # [torch.Size([16, 136800, 4]), torch.Size([16, 34200, 4]), torch.Size([16, 8550, 4]), torch.Size([16, 2223, 4]), torch.Size([16, 630, 4])]
    all_anchor_list = images_to_levels(concat_anchor_list,
                                           num_level_anchors)

    losses_cls, losses_bbox = multi_apply(
        self.loss_single,
        cls_scores,
        bbox_preds,
        all_anchor_list,
        labels_list,
        label_weights_list,
        bbox_targets_list,
        bbox_weights_list,
        num_total_samples=num_total_samples)
    return dict(loss_cls=losses_cls, loss_bbox=losses_bbox)
```





## loss_single



loss_single 以特征图为单位计算每个特征图上的 loss，然后再通过 `multi_apply` 汇总到 loss 函数里面得到最后的 loss

```python
def loss_single(self, cls_score, bbox_pred, anchors, labels, label_weights,
                bbox_targets, bbox_weights, num_total_samples):
    """Compute loss of a single scale level.

    Args:
        cls_score (Tensor): Box scores for each scale level
            Has shape (N, num_anchors * num_classes, H, W).
        bbox_pred (Tensor): Box energies / deltas for each scale
            level with shape (N, num_anchors * 4, H, W).
        anchors (Tensor): Box reference for each scale level with shape
            (N, num_total_anchors, 4).
        labels (Tensor): Labels of each anchors with shape
            (N, num_total_anchors).
        label_weights (Tensor): Label weights of each anchor with shape
            (N, num_total_anchors)
        bbox_targets (Tensor): BBox regression targets of each anchor wight
            shape (N, num_total_anchors, 4).
        bbox_weights (Tensor): BBox regression loss weights of each anchor
            with shape (N, num_total_anchors, 4).
        num_total_samples (int): If sampling, num total samples equal to
            the number of total anchors; Otherwise, it is the number of
            positive anchors.

    Returns:
        dict[str, Tensor]: A dictionary of loss components.
    """
    # classification loss
    # 正样本和负样本参与计算分类 loss
    labels = labels.reshape(-1)
    label_weights = label_weights.reshape(-1)
    cls_score = cls_score.permute(0, 2, 3,
                                  1).reshape(-1, self.cls_out_channels)
    loss_cls = self.loss_cls(
        cls_score, labels, label_weights, avg_factor=num_total_samples)
    # regression loss
    # 正样本参与计算回归 loss
    bbox_targets = bbox_targets.reshape(-1, 4)
    bbox_weights = bbox_weights.reshape(-1, 4)
    bbox_pred = bbox_pred.permute(0, 2, 3, 1).reshape(-1, 4)
    # 如果需要回归解码后的坐标的话，就根据当前位置的 anchor 解码出坐标，与 gt 做 loss
    # RetinaNet 里面默认是 False，注意解码之后
    if self.reg_decoded_bbox:
        # When the regression loss (e.g. `IouLoss`, `GIouLoss`)
        # is applied directly on the decoded bounding boxes, it
        # decodes the already encoded coordinates to absolute format.
        anchors = anchors.reshape(-1, 4)
        bbox_pred = self.bbox_coder.decode(anchors, bbox_pred)
    loss_bbox = self.loss_bbox(
        bbox_pred,
        bbox_targets,
        bbox_weights,
        avg_factor=num_total_samples)

    return loss_cls, loss_bbox
```

## get_anchors



这里通过每一层的 FPN 特征大小来生成每一层对应的 anchor，在 RetinaNet 中每一个像素点上是会生成 9 个大小不同的 anchor，最终返回一个 List[List[Tensor]]最外面 list 的 size 为 batch_size，里面的是 FPN 特征图的个数，最里面的 tensor 就是每一个特征图中所含有的 anchor 数目，shape 为 (A, 4)

```python
anchor_generator=dict(
    type='AnchorGenerator',
    octave_base_scale=4,
    scales_per_octave=3,
    ratios=[0.5, 1.0, 2.0],
    strides=[8, 16, 32, 64, 128]),
```



```python
def get_anchors(self, featmap_sizes, img_metas, device='cuda'):
    """Get anchors according to feature map sizes.

    Args:
        featmap_sizes (list[tuple]): Multi-level feature map sizes.
        img_metas (list[dict]): Image meta info.
        device (torch.device | str): Device for returned tensors

    Returns:
        tuple:
            anchor_list (list[Tensor]): Anchors of each image.
            valid_flag_list (list[Tensor]): Valid flags of each image.
    """
    num_imgs = len(img_metas)

    # since feature map sizes of all images are the same, we only compute
    # anchors for one time
    # 返回一个 list[Tensor]，每一个元素就是当前尺度特征图下的所有 anchor 的数量，shape: (HWA,4)
	# [torch.Size([136800, 4]), torch.Size([34200, 4]), torch.Size([8550, 4]), torch.Size([2223, 4]), torch.Size([630, 4])]
    multi_level_anchors = self.anchor_generator.grid_anchors(
        featmap_sizes, device)
    # 复制了 batch_size 次，因为在一个 batch 中每一张图片的 size 都是一样的，所以生成的 anchor 也是一样的
    anchor_list = [multi_level_anchors for _ in range(num_imgs)]

    # for each image, we compute valid flags of multi level anchors
    valid_flag_list = []
    for img_id, img_meta in enumerate(img_metas):
        multi_level_flags = self.anchor_generator.valid_flags(
            featmap_sizes, img_meta['pad_shape'], device)
        valid_flag_list.append(multi_level_flags)

    return anchor_list, valid_flag_list
```





## grid_anchors



这个函数在 `mmdet/core/anchor/anchor_generator.py` 中，作用就是通过所有尺度的特征图和 base_anchor 的偏移来生成一张图片中所有 anchor，返回的是一个 list，每一个元素就是一个尺度的特征图中所有的 anchor

```python
def grid_anchors(self, featmap_sizes, device='cuda'):
    """Generate grid anchors in multiple feature levels.

    Args:
        featmap_sizes (list[tuple]): List of feature map sizes in
            multiple feature levels.
        device (str): Device where the anchors will be put on.

    Return:
        list[torch.Tensor]: Anchors in multiple feature levels. \
            The sizes of each tensor should be [N, 4], where \
            N = width * height * num_base_anchors, width and height \
            are the sizes of the corresponding feature level, \
            num_base_anchors is the number of anchors for that level.
    """
    assert self.num_levels == len(featmap_sizes)
    multi_level_anchors = []
    for i in range(self.num_levels):
        anchors = self.single_level_grid_anchors(
            self.base_anchors[i].to(device),
            featmap_sizes[i],
            self.strides[i],
            device=device)
        multi_level_anchors.append(anchors)
    return multi_level_anchors
```





## self.sampling



指的是是否要在所有 anchor 样本中取样一部分进行计算 loss，避免负样本太多对模型造成损害，用 FocalLoss 之类的话就不用 sampling

```python
self.sampling = loss_cls['type'] not in [
    'FocalLoss', 'GHMC', 'QualityFocalLoss'
]
```



刚刚看了一下最新版本的，这边已经替换了，换了个更好的方式

```python
if self.train_cfg:
    self.assigner = build_assigner(self.train_cfg.assigner)
    if hasattr(self.train_cfg,
               'sampler') and self.train_cfg.sampler.type.split(
        '.')[-1] != 'PseudoSampler':
        self.sampling = True
        sampler_cfg = self.train_cfg.sampler
        # avoid BC-breaking
        if loss_cls['type'] in [
            'FocalLoss', 'GHMC', 'QualityFocalLoss'
        ]:
            warnings.warn(
                'DeprecationWarning: Determining whether to sampling'
                'by loss type is deprecated, please delete sampler in'
                'your config when using `FocalLoss`, `GHMC`, '
                '`QualityFocalLoss` or other FocalLoss variant.')
            self.sampling = False
            sampler_cfg = dict(type='PseudoSampler')
            else:
                self.sampling = False
                sampler_cfg = dict(type='PseudoSampler')
                self.sampler = build_sampler(sampler_cfg, context=self)
```



## image_to_levels



这个函数将所有图片上的东西转化成针对每一个特征图级别的东西，比如 anchor，我们传进来的 target 参数就是一个长度为 batch_size 的列表，列表每一个元素都是一样的，shape 是 (all_anchors_in_a_img, 4)，经过 stack 之后 target 就成了 shape(batch_size, all_anchors_in_a_img, 4) 的 tensor，然后再根据每一层的 anchor 数将其分割出来返回



```python
def images_to_levels(target, num_levels):
    """Convert targets by image to targets by feature level.

    [target_img0, target_img1] -> [target_level0, target_level1, ...]
    """
    target = torch.stack(target, 0)
    level_targets = []
    start = 0
    for n in num_levels:
        end = start + n
        # level_targets.append(target[:, start:end].squeeze(0))
        level_targets.append(target[:, start:end])
        start = end
    return level_targets
```



## get_targets



get_targets 是根据每一张图片来获取标签的，所以这里传入的参数不需要进行 `image_to_level` 操作，但是在函数内部，由于计算 loss 是针对每一个特征图来计算 loss 的，所以在内部会用 `image_to_level` 函数将标签转化成每一个特征图级别的标签，具体看下面解释，传入的参数是针对每一张图片的，返回的东西都是针对每一个特征图的，要注意这种差别，很容易疏忽！



```python
def get_targets(self,
                anchor_list,
                valid_flag_list,
                gt_bboxes_list,
                img_metas,
                gt_bboxes_ignore_list=None,
                gt_labels_list=None,
                label_channels=1,
                unmap_outputs=True,
                return_sampling_results=False):                             
    	"""Compute regression and classification targets for anchors in
        multiple images.

        Args:
            anchor_list (list[list[Tensor]]): Multi level anchors of each
                image. The outer list indicates images, and the inner list
                corresponds to feature levels of the image. Each element of
                the inner list is a tensor of shape (num_anchors, 4).
            valid_flag_list (list[list[Tensor]]): Multi level valid flags of
                each image. The outer list indicates images, and the inner list
                corresponds to feature levels of the image. Each element of
                the inner list is a tensor of shape (num_anchors, )
            gt_bboxes_list (list[Tensor]): Ground truth bboxes of each image.
            img_metas (list[dict]): Meta info of each image.
            gt_bboxes_ignore_list (list[Tensor]): Ground truth bboxes to be
                ignored.
            gt_labels_list (list[Tensor]): Ground truth labels of each box.
            label_channels (int): Channel of label.
            unmap_outputs (bool): Whether to map outputs back to the original
                set of anchors.

        Returns:
            tuple: Usually returns a tuple containing learning targets.

                - labels_list (list[Tensor]): Labels of each level.
                - label_weights_list (list[Tensor]): Label weights of each \
                    level.
                - bbox_targets_list (list[Tensor]): BBox targets of each level.
                - bbox_weights_list (list[Tensor]): BBox weights of each level.
                - num_total_pos (int): Number of positive samples in all \
                    images.
                - num_total_neg (int): Number of negative samples in all \
                    images.
            additional_returns: This function enables user-defined returns from
                `self._get_targets_single`. These returns are currently refined
                to properties at each feature map (i.e. having HxW dimension).
                The results will be concatenated after the end
                """    
        
num_imgs = len(img_metas)
assert len(anchor_list) == len(valid_flag_list) == num_imgs

# anchor number of multi levels
num_level_anchors = [anchors.size(0) for anchors in anchor_list[0]]
# concat all level anchors to a single tensor
concat_anchor_list = []
concat_valid_flag_list = []
# 将一张图上的所有 anchor 全部 concat 在一起
# concat_anchor_list 此时是 [torch.Size([182403, 4])]*batch_size 的一个列表
for i in range(num_imgs):
    assert len(anchor_list[i]) == len(valid_flag_list[i])
    concat_anchor_list.append(torch.cat(anchor_list[i]))
    concat_valid_flag_list.append(torch.cat(valid_flag_list[i]))

# compute targets for each image
if gt_bboxes_ignore_list is None:
    gt_bboxes_ignore_list = [None for _ in range(num_imgs)]
if gt_labels_list is None:
    gt_labels_list = [None for _ in range(num_imgs)]

# 调用 _get_target_single 得到每一张图的标签
# multi_apply 返回一个 tuple，results 中很多元素都是 list，
# list 中每个元素代表了该图片中的一些信息
results = multi_apply(
    self._get_targets_single,
    concat_anchor_list,
    concat_valid_flag_list,
    gt_bboxes_list,
    gt_bboxes_ignore_list,
    gt_labels_list,
    img_metas,
    label_channels=label_channels,
    unmap_outputs=unmap_outputs)
(all_labels, all_label_weights, all_bbox_targets, all_bbox_weights,
 	pos_inds_list, neg_inds_list, sampling_results_list) = results[:7]
rest_results = list(results[7:])  # user-added return values
# no valid anchors
if any([labels is None for labels in all_labels]):
    return None
# sampled anchors of all images
# 得到一个 batch 中所有的正样本数目和负样本数目
num_total_pos = sum([max(inds.numel(), 1) for inds in pos_inds_list])
num_total_neg = sum([max(inds.numel(), 1) for inds in neg_inds_list])
# split targets to a list w.r.t. multiple levels
# 然后用 image_to_level 根据每一层特征图上 anchor 的数量将图片级别的标签转成特征图级别的标签
# labels_list: List[Tensor]
# [torch.Size([16, 136800]), torch.Size([16, 34200]), torch.Size([16, 8550]), torch.Size([16, 2223]), torch.Size([16, 630])]
labels_list = images_to_levels(all_labels, num_level_anchors)
label_weights_list = images_to_levels(all_label_weights,
                                      num_level_anchors)
bbox_targets_list = images_to_levels(all_bbox_targets,
                                     num_level_anchors)
bbox_weights_list = images_to_levels(all_bbox_weights,
                                     num_level_anchors)
res = (labels_list, label_weights_list, bbox_targets_list,
       bbox_weights_list, num_total_pos, num_total_neg)
# sampling_results_list 包含了每张图中的正负样本信息等等
if return_sampling_results:
    res = res + (sampling_results_list, )
for i, r in enumerate(rest_results):  # user-added return values
    rest_results[i] = images_to_levels(r, num_level_anchors)

return res + tuple(rest_results)        
```



## _get_targets_single



这个函数接收一张图里面的所有 anchor 信息，开始和 gt_bboxes 进行匹配，得到每一个 anchor 需要回归的目标

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
    """Compute regression and classification targets for anchors in a
    single image.

    Args:
        flat_anchors (Tensor): Multi-level anchors of the image, which are
            concatenated into a single tensor of shape (num_anchors ,4)
        valid_flags (Tensor): Multi level valid flags of the image,
            which are concatenated into a single tensor of
                shape (num_anchors,).
        gt_bboxes (Tensor): Ground truth bboxes of the image,
            shape (num_gts, 4).
        gt_bboxes_ignore (Tensor): Ground truth bboxes to be
            ignored, shape (num_ignored_gts, 4).
        img_meta (dict): Meta info of the image.
        gt_labels (Tensor): Ground truth labels of each box,
            shape (num_gts,).
        label_channels (int): Channel of label.
        unmap_outputs (bool): Whether to map outputs back to the original
            set of anchors.

    Returns:
        tuple:
            labels_list (list[Tensor]): Labels of each level
            label_weights_list (list[Tensor]): Label weights of each level
            bbox_targets_list (list[Tensor]): BBox targets of each level
            bbox_weights_list (list[Tensor]): BBox weights of each level
            num_total_pos (int): Number of positive samples in all images
            num_total_neg (int): Number of negative samples in all images
    """
    # 判断 anchor 是否越界
    # ipdb> flat_anchors.shape
	# torch.Size([182403, 4])
    # ipdb> anchors.shape
	# torch.Size([163206, 4])
    inside_flags = anchor_inside_flags(flat_anchors, valid_flags,
                                       img_meta['img_shape'][:2],
                                       self.train_cfg.allowed_border)
    if not inside_flags.any():
        return (None, ) * 7
    # assign gt and sample anchors
    # 这里会筛掉一些越界的 anchor 
    # 关系式：flat_anchors = anchors + 越界 anchors
    # anchors = pos_anchors + neg_anchors
    anchors = flat_anchors[inside_flags, :]

    # 对 anchor 和 gt_bboxes 进行一一匹配得到 sample
    # 得到正负样本，每一个 anchor 匹配的 gt index 
    assign_result = self.assigner.assign(
        anchors, gt_bboxes, gt_bboxes_ignore,
        None if self.sampling else gt_labels)
    # 对结果进行包装，如果没有 self.sampler 的话就是用 PseudoSampler
    sampling_result = self.sampler.sample(assign_result, anchors,
                                          gt_bboxes)
	
    # 这张图片中没有越界的 anchor 的数量
    num_valid_anchors = anchors.shape[0]
    bbox_targets = torch.zeros_like(anchors)
    bbox_weights = torch.zeros_like(anchors)
    # 首先将每个 anchor 的类别都置为背景，权重置为 0 
    labels = anchors.new_full((num_valid_anchors, ),
                              self.num_classes,
                              dtype=torch.long)
    label_weights = anchors.new_zeros(num_valid_anchors, dtype=torch.float)
	
    # 得到正负样本的 index （负样本为 IOU>=0 并且 < 0.4，）
    pos_inds = sampling_result.pos_inds
    neg_inds = sampling_result.neg_inds
    if len(pos_inds) > 0:
        # 如果回归的坐标不是解码过后的坐标（format: x1,y1,x2,y2）
        # 就对正样本 anchor 的坐标和 gt_bboxes 进行编码 normalize
        if not self.reg_decoded_bbox:
            pos_bbox_targets = self.bbox_coder.encode(
                # 这两个指的是正样本 anchor 的坐标，正样本匹配的（需要回归的）gt_bbox 的坐标
                # shape 都是 [pos_anchors, 4]
                # 如果 encode 的话会将 pos_bbox_targets 的值 normalize 到（-1，1）之间，所以最小的值不是 0
                sampling_result.pos_bboxes, sampling_result.pos_gt_bboxes)
        else:
            # 回归解码坐标的话，targets 就是每个 anchor 匹配的 gt_bbox 坐标
            pos_bbox_targets = sampling_result.pos_gt_bboxes
        # 给正样本回归目标赋值
        bbox_targets[pos_inds, :] = pos_bbox_targets
        # 给正样本回归权重赋值，所以只有正样本参与计算 bbox_loss
        # 负样本和越界的 anchor 都没有参加，因为权重是 0
        bbox_weights[pos_inds, :] = 1.0
        if gt_labels is None:
            # Only rpn gives gt_labels as None
            # Foreground is the first class since v2.5.0
            labels[pos_inds] = 0
        else:
            # 给正样本分类目标赋值
            labels[pos_inds] = gt_labels[
                sampling_result.pos_assigned_gt_inds]
        if self.train_cfg.pos_weight <= 0:
            # 给正样本分类权重赋值
            label_weights[pos_inds] = 1.0
        else:
            label_weights[pos_inds] = self.train_cfg.pos_weight
    if len(neg_inds) > 0:
        # 给负样本分类权重赋值，所以正负样本都会参与计算 loss
        # 但是越界的那些 anchor 就没有参加，因为它们的 loss 权重是 0
        label_weights[neg_inds] = 1.0

    # map up to original set of anchors
    if unmap_outputs:
        # 给越界的 anchor 也贴上标签，分类标签赋值为背景，回归标签赋值为 0
        # 这样的话每次 get_targets 得到的 anchor 数量还是一样的
        num_total_anchors = flat_anchors.size(0)
        labels = unmap(
            labels, num_total_anchors, inside_flags,
            fill=self.num_classes)  # fill bg label
        label_weights = unmap(label_weights, num_total_anchors,
                              inside_flags)
        bbox_targets = unmap(bbox_targets, num_total_anchors, inside_flags)
        bbox_weights = unmap(bbox_weights, num_total_anchors, inside_flags)

    return (labels, label_weights, bbox_targets, bbox_weights, pos_inds,
            neg_inds, sampling_result)
```





## anchor_inside_flags



这个函数判断一个 anchor box 是否在图像范围内，有意思的是这里 img_shape 用的是 `img_meta['img_shape'][:2]`,在我们这里是 `800*800`，但是我们一个 batch 中拿去训练的图像的尺寸为 `1216*800`，我大概猜一下，其中有 `416*800` 的部分是被 padding 的，所以在这部分区域生成的 anchor 都不是 valid 的

```python
inside_flags = anchor_inside_flags(flat_anchors, valid_flags,
                                           img_meta['img_shape'][:2],
                                           self.train_cfg.allowed_border)
```



```python
def anchor_inside_flags(flat_anchors,
                        valid_flags,
                        img_shape,
                        allowed_border=0):
    """Check whether the anchors are inside the border.

    Args:
        flat_anchors (torch.Tensor): Flatten anchors, shape (n, 4).
        valid_flags (torch.Tensor): An existing valid flags of anchors.
        img_shape (tuple(int)): Shape of current image.
        allowed_border (int, optional): The border to allow the valid anchor.
            Defaults to 0.

    Returns:
        torch.Tensor: Flags indicating whether the anchors are inside a \
            valid range.
    """
    img_h, img_w = img_shape[:2]
    if allowed_border >= 0:
        inside_flags = valid_flags & \
            (flat_anchors[:, 0] >= -allowed_border) & \
            (flat_anchors[:, 1] >= -allowed_border) & \
            (flat_anchors[:, 2] < img_w + allowed_border) & \
            (flat_anchors[:, 3] < img_h + allowed_border)
    else:
        inside_flags = valid_flags
    return inside_flags
```



## assigner.assign



这个函数起的作用就是根据一些标准对 anchor 和 gt_bboxes 进行匹配得到正负样本或者忽略样本。RetinaNet 选用的是一般的 `MaxIoUAssigner`，是根据 anchor 和 gt_bboxes 的 IoU 进行匹配的

```python
assigner=dict(
    type='MaxIoUAssigner',
    pos_iou_thr=0.5,
    neg_iou_thr=0.4,
    min_pos_iou=0,
    ignore_iof_thr=-1),
allowed_border=-1,
pos_weight=-1,
debug=False),
```



```python
def assign(self, bboxes, gt_bboxes, gt_bboxes_ignore=None, gt_labels=None):
    """Assign gt to bboxes.

    This method assign a gt bbox to every bbox (proposal/anchor), each bbox
    will be assigned with -1, or a semi-positive number. -1 means negative
    sample, semi-positive number is the index (0-based) of assigned gt.
    The assignment is done in following steps, the order matters.

    1. assign every bbox to the background
    2. assign proposals whose iou with all gts < neg_iou_thr to 0
    3. for each bbox, if the iou with its nearest gt >= pos_iou_thr,
       assign it to that bbox
    4. for each gt bbox, assign its nearest proposals (may be more than
       one) to itself

    Args:
        bboxes (Tensor): Bounding boxes to be assigned, shape(n, 4).
        gt_bboxes (Tensor): Groundtruth boxes, shape (k, 4).
        gt_bboxes_ignore (Tensor, optional): Ground truth bboxes that are
            labelled as `ignored`, e.g., crowd boxes in COCO.
        gt_labels (Tensor, optional): Label of gt_bboxes, shape (k, ).

    Returns:
        :obj:`AssignResult`: The assign result.

    Example:
        >>> self = MaxIoUAssigner(0.5, 0.5)
        >>> bboxes = torch.Tensor([[0, 0, 10, 10], [10, 10, 20, 20]])
        >>> gt_bboxes = torch.Tensor([[0, 0, 10, 9]])
        >>> assign_result = self.assign(bboxes, gt_bboxes)
        >>> expected_gt_inds = torch.LongTensor([1, 0])
        >>> assert torch.all(assign_result.gt_inds == expected_gt_inds)
    """
    assign_on_cpu = True if (self.gpu_assign_thr > 0) and (
        gt_bboxes.shape[0] > self.gpu_assign_thr) else False
    # compute overlap and assign gt on CPU when number of GT is large
    if assign_on_cpu:
        device = bboxes.device
        bboxes = bboxes.cpu()
        gt_bboxes = gt_bboxes.cpu()
        if gt_bboxes_ignore is not None:
            gt_bboxes_ignore = gt_bboxes_ignore.cpu()
        if gt_labels is not None:
            gt_labels = gt_labels.cpu()
            
	# 将 gt_bboxes 和 这张图里面所有的 anchor 进行两两匹配，shape: (num_gt, num_anchor)
    overlaps = self.iou_calculator(gt_bboxes, bboxes)

    if (self.ignore_iof_thr > 0 and gt_bboxes_ignore is not None
            and gt_bboxes_ignore.numel() > 0 and bboxes.numel() > 0):
        if self.ignore_wrt_candidates:
            ignore_overlaps = self.iou_calculator(
                bboxes, gt_bboxes_ignore, mode='iof')
            ignore_max_overlaps, _ = ignore_overlaps.max(dim=1)
        else:
            ignore_overlaps = self.iou_calculator(
                gt_bboxes_ignore, bboxes, mode='iof')
            ignore_max_overlaps, _ = ignore_overlaps.max(dim=0)
        overlaps[:, ignore_max_overlaps > self.ignore_iof_thr] = -1
	
    # 根据匹配结果进行包装，同时给 anchor 添加上类别标签
    assign_result = self.assign_wrt_overlaps(overlaps, gt_labels)
    if assign_on_cpu:
        assign_result.gt_inds = assign_result.gt_inds.to(device)
        assign_result.max_overlaps = assign_result.max_overlaps.to(device)
        if assign_result.labels is not None:
            assign_result.labels = assign_result.labels.to(device)
    return assign_result
```



## assigner.assign_wrt_overlaps



这个函数其实起到的是一个包装的作用，在上面已经将 anchor 和 gt 进行了两两匹配，输入的 overlaps 参数的形状就是 (num_gt, num_anchor)，同时也将 gt_bboxes 的类别标签传了进来，



```python
def assign_wrt_overlaps(self, overlaps, gt_labels=None):
    """Assign w.r.t. the overlaps of bboxes with gts.

    Args:
        overlaps (Tensor): Overlaps between k gt_bboxes and n bboxes,
            shape(k, n).
        gt_labels (Tensor, optional): Labels of k gt_bboxes, shape (k, ).

    Returns:
        :obj:`AssignResult`: The assign result.
    """
    num_gts, num_bboxes = overlaps.size(0), overlaps.size(1)

    # 1. assign -1 by default
    # 第一步，给每一个 anchor box 的匹配的 gt index 都初始化为 -1    
    # shape：torch.Size([163206])
    assigned_gt_inds = overlaps.new_full((num_bboxes, ),
                                         -1,
                                         dtype=torch.long)

    if num_gts == 0 or num_bboxes == 0:
        # No ground truth or boxes, return empty assignment
        max_overlaps = overlaps.new_zeros((num_bboxes, ))
        if num_gts == 0:
            # No truth, assign everything to background
            assigned_gt_inds[:] = 0
        if gt_labels is None:
            assigned_labels = None
        else:
            assigned_labels = overlaps.new_full((num_bboxes, ),
                                                -1,
                                                dtype=torch.long)
        return AssignResult(
            num_gts,
            assigned_gt_inds,
            max_overlaps,
            labels=assigned_labels)

    # for each anchor, which gt best overlaps with it
    # for each anchor, the max iou of all gts
    # 找到每一个 anchor 对应的最大 IoU 的 gt_bboxes
    # 记录下每一个 anchor 最大的 IoU 和对应 gt_bboxes 的 index
    max_overlaps, argmax_overlaps = overlaps.max(dim=0)
    # for each gt, which anchor best overlaps with it
    # for each gt, the max iou of all proposals
    # 记录下每一个 gt 和所有 anchor 最大的 IoU 和对应 anchor 的 index
    gt_max_overlaps, gt_argmax_overlaps = overlaps.max(dim=1)

    # 2. assign negative: below
    # the negative inds are set to be 0
    # 匹配 0 的话就是负样本
    if isinstance(self.neg_iou_thr, float):
        assigned_gt_inds[(max_overlaps >= 0)
                         & (max_overlaps < self.neg_iou_thr)] = 0
    elif isinstance(self.neg_iou_thr, tuple):
        assert len(self.neg_iou_thr) == 2
        assigned_gt_inds[(max_overlaps >= self.neg_iou_thr[0])
                         & (max_overlaps < self.neg_iou_thr[1])] = 0

    # 3. assign positive: above positive IoU threshold
    pos_inds = max_overlaps >= self.pos_iou_thr
    # 这里的 +1 是一个 trick，意义上来说的话是所有 positive anchor 匹配第几个 gt
    # +1 就把这些值全部搞成正的了，方便后面根据匹配的是不是大于 0 对负样本和背景进行筛除
    assigned_gt_inds[pos_inds] = argmax_overlaps[pos_inds] + 1

    if self.match_low_quality:
        # Low-quality matching will overwrite the assigned_gt_inds assigned
        # in Step 3. Thus, the assigned gt might not be the best one for
        # prediction.
        # For example, if bbox A has 0.9 and 0.8 iou with GT bbox 1 & 2,
        # bbox 1 will be assigned as the best target for bbox A in step 3.
        # However, if GT bbox 2's gt_argmax_overlaps = A, bbox A's
        # assigned_gt_inds will be overwritten to be bbox B.
        # This might be the reason that it is not used in ROI Heads.
        for i in range(num_gts):
            if gt_max_overlaps[i] >= self.min_pos_iou:
                if self.gt_max_assign_all:
                    max_iou_inds = overlaps[i, :] == gt_max_overlaps[i]
                    assigned_gt_inds[max_iou_inds] = i + 1
                else:
                    assigned_gt_inds[gt_argmax_overlaps[i]] = i + 1

    if gt_labels is not None:
        # 默认初始化每一个 anchor 的类别为 -1
        assigned_labels = assigned_gt_inds.new_full((num_bboxes, ), -1)
        # 这里将所有的正样本的 index 给找了出来
        pos_inds = torch.nonzero(
            assigned_gt_inds > 0, as_tuple=False).squeeze()        
        if pos_inds.numel() > 0:
            # 给每一个正样本都赋予了类别标签，上面 +1，这里 -1，刚好得到 gt 标签
            # ipdb> assigned_labels.shape
			# torch.Size([163206])
            # ipdb> assigned_labels.unique()
			# tensor([-1,  0, 48, 52], device='cuda:0')
			# -1 代表负样本，0 和 0 以上的都是真实标签
            assigned_labels[pos_inds] = gt_labels[
                assigned_gt_inds[pos_inds] - 1]
    else:
        assigned_labels = None

    return AssignResult(
        num_gts, assigned_gt_inds, max_overlaps, labels=assigned_labels)
# class AssignResult(util_mixins.NiceRepr):
#     def __init__(self, num_gts, gt_inds, max_overlaps, labels=None):
```



## AssignResult



这个就是储存了一下 assign 的结果，方便调用，没有什么实际作用

```python
class AssignResult(util_mixins.NiceRepr):
    """Stores assignments between predicted and truth boxes.

    Attributes:
        num_gts (int): the number of truth boxes considered when computing this
            assignment

        gt_inds (LongTensor): for each predicted box indicates the 1-based
            index of the assigned truth box. 0 means unassigned and -1 means
            ignore.

        max_overlaps (FloatTensor): the iou between the predicted box and its
            assigned truth box.

        labels (None | LongTensor): If specified, for each predicted box
            indicates the category label of the assigned truth box.

    Example:
        >>> # An assign result between 4 predicted boxes and 9 true boxes
        >>> # where only two boxes were assigned.
        >>> num_gts = 9
        >>> max_overlaps = torch.LongTensor([0, .5, .9, 0])
        >>> gt_inds = torch.LongTensor([-1, 1, 2, 0])
        >>> labels = torch.LongTensor([0, 3, 4, 0])
        >>> self = AssignResult(num_gts, gt_inds, max_overlaps, labels)
        >>> print(str(self))  # xdoctest: +IGNORE_WANT
        <AssignResult(num_gts=9, gt_inds.shape=(4,), max_overlaps.shape=(4,),
                      labels.shape=(4,))>
        >>> # Force addition of gt labels (when adding gt as proposals)
        >>> new_labels = torch.LongTensor([3, 4, 5])
        >>> self.add_gt_(new_labels)
        >>> print(str(self))  # xdoctest: +IGNORE_WANT
        <AssignResult(num_gts=9, gt_inds.shape=(7,), max_overlaps.shape=(7,),
                      labels.shape=(7,))>
    """

    def __init__(self, num_gts, gt_inds, max_overlaps, labels=None):
        self.num_gts = num_gts
        self.gt_inds = gt_inds
        self.max_overlaps = max_overlaps
        self.labels = labels
        # Interface for possible user-defined properties
        self._extra_properties = {}
```



## iou_calculator



计算两批 box 之间的 IoU，传进去的 box 的坐标都要是绝对坐标 (format: x1, y1, x2, y2)，不能够是归一化后的坐标，其中还有个 `is_aligned` 参数，默认是 False，也就是说这两批 box 两两都要求一次 IoU，最终输出的维度就是 (num_box1, num_box2)。 如果是 True 的话，这两批 box 的个数必须相等，在对应的 index 上求两个 box 之间的 IoU，不用每一个都求。

```python
@IOU_CALCULATORS.register_module()
class BboxOverlaps2D(object):
    """2D Overlaps (e.g. IoUs, GIoUs) Calculator."""

    def __call__(self, bboxes1, bboxes2, mode='iou', is_aligned=False):
        """Calculate IoU between 2D bboxes.

        Args:
            bboxes1 (Tensor): bboxes have shape (m, 4) in <x1, y1, x2, y2>
                format, or shape (m, 5) in <x1, y1, x2, y2, score> format.
            bboxes2 (Tensor): bboxes have shape (m, 4) in <x1, y1, x2, y2>
                format, shape (m, 5) in <x1, y1, x2, y2, score> format, or be
                empty. If ``is_aligned `` is ``True``, then m and n must be
                equal.
            mode (str): "iou" (intersection over union), "iof" (intersection
                over foreground), or "giou" (generalized intersection over
                union).
            is_aligned (bool, optional): If True, then m and n must be equal.
                Default False.

        Returns:
            Tensor: shape (m, n) if ``is_aligned `` is False else shape (m,)
        """
        assert bboxes1.size(-1) in [0, 4, 5]
        assert bboxes2.size(-1) in [0, 4, 5]
        if bboxes2.size(-1) == 5:
            bboxes2 = bboxes2[..., :4]
        if bboxes1.size(-1) == 5:
            bboxes1 = bboxes1[..., :4]
        return bbox_overlaps(bboxes1, bboxes2, mode, is_aligned)
```





## sampler.sample



RetinaNet 中并没有是用 sampling ，为了让接口统一，所以就用了一个假的 Sampler 代替，其实没有改变 assign_result 的结果

```python
@BBOX_SAMPLERS.register_module()
class PseudoSampler(BaseSampler):
    """A pseudo sampler that does not do sampling actually."""

    def __init__(self, **kwargs):
        pass
    
    def sample(self, assign_result, bboxes, gt_bboxes, **kwargs):
        """Directly returns the positive and negative indices  of samples.

        Args:
            assign_result (:obj:`AssignResult`): Assigned results
            bboxes (torch.Tensor): Bounding boxes
            gt_bboxes (torch.Tensor): Ground truth boxes

        Returns:
            :obj:`SamplingResult`: sampler results
        """
        # 得到正负样本 anchor 的 index
        pos_inds = torch.nonzero(
            assign_result.gt_inds > 0, as_tuple=False).squeeze(-1).unique()
        neg_inds = torch.nonzero(
            assign_result.gt_inds == 0, as_tuple=False).squeeze(-1).unique()
        gt_flags = bboxes.new_zeros(bboxes.shape[0], dtype=torch.uint8)
        # 进行封装，方便调用 sampling_result
        sampling_result = SamplingResult(pos_inds, neg_inds, bboxes, gt_bboxes,
                                         assign_result, gt_flags)
        return sampling_result
```



## SamplingResult



进行封装 sampling 的结果



```python
class SamplingResult(util_mixins.NiceRepr):
    """Bbox sampling result.

    Example:
        >>> # xdoctest: +IGNORE_WANT
        >>> from mmdet.core.bbox.samplers.sampling_result import *  # NOQA
        >>> self = SamplingResult.random(rng=10)
        >>> print(f'self = {self}')
        self = <SamplingResult({
            'neg_bboxes': torch.Size([12, 4]),
            'neg_inds': tensor([ 0,  1,  2,  4,  5,  6,  7,  8,  9, 10, 11, 12]),
            'num_gts': 4,
            'pos_assigned_gt_inds': tensor([], dtype=torch.int64),
            'pos_bboxes': torch.Size([0, 4]),
            'pos_inds': tensor([], dtype=torch.int64),
            'pos_is_gt': tensor([], dtype=torch.uint8)
        })>
    """

    def __init__(self, pos_inds, neg_inds, bboxes, gt_bboxes, assign_result,
                 gt_flags):
        self.pos_inds = pos_inds
        self.neg_inds = neg_inds
        self.pos_bboxes = bboxes[pos_inds]
        self.neg_bboxes = bboxes[neg_inds]
        self.pos_is_gt = gt_flags[pos_inds]

        self.num_gts = gt_bboxes.shape[0]
        self.pos_assigned_gt_inds = assign_result.gt_inds[pos_inds] - 1

        if gt_bboxes.numel() == 0:
            # hack for index error case
            assert self.pos_assigned_gt_inds.numel() == 0
            self.pos_gt_bboxes = torch.empty_like(gt_bboxes).view(-1, 4)
        else:
            if len(gt_bboxes.shape) < 2:
                gt_bboxes = gt_bboxes.view(-1, 4)

            self.pos_gt_bboxes = gt_bboxes[self.pos_assigned_gt_inds, :]

        if assign_result.labels is not None:
            self.pos_gt_labels = assign_result.labels[pos_inds]
        else:
            self.pos_gt_labels = None
```





## bbox_coder.encode & bbox2delta



这个是从 Faster RCNN 开始就祖传的 box 编码，RetinaNet 也继续沿用 `DeltaXYWHBBoxCoder` 编码器，也就是说网络学习的目标为正样本到 gt_bbox 所需要的偏移量

```python
bbox_coder=dict(
    type='DeltaXYWHBBoxCoder',
    target_means=[.0, .0, .0, .0],
    target_stds=[1.0, 1.0, 1.0, 1.0]),
```



所以 encode 函数就将 anchor 和 匹配的 gt_bbox 的差距进行编码，让网络学习，实际上调用的就是 `bbox2delta` 这个函数

```python
def encode(self, bboxes, gt_bboxes):
    """Get box regression transformation deltas that can be used to
    transform the ``bboxes`` into the ``gt_bboxes``.

    Args:
        bboxes (torch.Tensor): Source boxes, e.g., object proposals.
        gt_bboxes (torch.Tensor): Target of the transformation, e.g.,
            ground-truth boxes.

    Returns:
        torch.Tensor: Box transformation deltas
    """

    assert bboxes.size(0) == gt_bboxes.size(0)
    assert bboxes.size(-1) == gt_bboxes.size(-1) == 4
    encoded_bboxes = bbox2delta(bboxes, gt_bboxes, self.means, self.stds)
    return encoded_bboxes


@mmcv.jit(coderize=True)
def bbox2delta(proposals, gt, means=(0., 0., 0., 0.), stds=(1., 1., 1., 1.)):
    """Compute deltas of proposals w.r.t. gt.

    We usually compute the deltas of x, y, w, h of proposals w.r.t ground
    truth bboxes to get regression target.
    This is the inverse function of :func:`delta2bbox`.

    Args:
        proposals (Tensor): Boxes to be transformed, shape (N, ..., 4)
        gt (Tensor): Gt bboxes to be used as base, shape (N, ..., 4)
        means (Sequence[float]): Denormalizing means for delta coordinates
        stds (Sequence[float]): Denormalizing standard deviation for delta
            coordinates

    Returns:
        Tensor: deltas with shape (N, 4), where columns represent dx, dy,
            dw, dh.
    """
    assert proposals.size() == gt.size()

    proposals = proposals.float()
    gt = gt.float()
    # 得到 anchor 的中心点和宽高
    px = (proposals[..., 0] + proposals[..., 2]) * 0.5
    py = (proposals[..., 1] + proposals[..., 3]) * 0.5
    pw = proposals[..., 2] - proposals[..., 0]
    ph = proposals[..., 3] - proposals[..., 1]
    
    # 得到 gt_bboxes 的中心点和宽高
    gx = (gt[..., 0] + gt[..., 2]) * 0.5
    gy = (gt[..., 1] + gt[..., 3]) * 0.5
    gw = gt[..., 2] - gt[..., 0]
    gh = gt[..., 3] - gt[..., 1]
	
    # 进行编码
    dx = (gx - px) / pw
    dy = (gy - py) / ph
    dw = torch.log(gw / pw)
    dh = torch.log(gh / ph)
    deltas = torch.stack([dx, dy, dw, dh], dim=-1)

    means = deltas.new_tensor(means).unsqueeze(0)
    stds = deltas.new_tensor(stds).unsqueeze(0)
    deltas = deltas.sub_(means).div_(stds)

    return deltas
```



## bbox_coder.decode & delta2bbox



刚好跟 encode 的过程是相反的，根据预测值与对应的 anchor 的坐标解得预测框的真实坐标，这里编码解码的方式可以自定义，但是一定要统一，编码时候对 anchor 怎样操作的解码时就要是一个逆过程



## umap



这个函数的作用就是将筛除过的正常 anchor 和被筛除的越界的 anchor 的标签汇合，在数量上变成原来的所有的 anchor，这里就要给这些越界的 anchor 分配背景标签，回归的 box 坐标也全都是 0

```python
def unmap(data, count, inds, fill=0):
    """Unmap a subset of item (data) back to the original set of items (of size
    count)"""
    if data.dim() == 1:
        ret = data.new_full((count, ), fill)
        ret[inds.type(torch.bool)] = data
    else:
        # new_size 这个表达妙！
        new_size = (count, ) + data.size()[1:]
        ret = data.new_full(new_size, fill)
        ret[inds.type(torch.bool), :] = data
    return ret
```





## 完结，撒花

