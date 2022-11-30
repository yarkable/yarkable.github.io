---
layout: post
title: mmdetection之DETR注释详解
subtitle: 
date: 2022-03-30
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



本文记录 mmdetection 对 DETR 训练的流程，包括标签获取，transformer encoder&decoder，前向训练，以及各步骤中 tensor 的形状，仅供复习用处。mmdetection 版本为 2.11.0。



## DETR



先从整个模型的 detector 看起，DETR 直接继承了 `SingleStageDetector`，所以改变的就是检测头，重点都在 TransformerHead 里面，我们直接从 forward_train 开始看



## TransformerHead



### forward_train



跟其他的检测头差不多，先是调用自己，也就是自身的 forward 函数，得到输出的 class label 和 reg coordinate，再调用自身的 loss 函数，不过这里是重载了一下，将 img_meta 传输进了 forward 函数的参数。

```python
# over-write because img_metas are needed as inputs for bbox_head.
def forward_train(self,
                    x,
                    img_metas,
                    gt_bboxes,
                    gt_labels=None,
                    gt_bboxes_ignore=None,
                    proposal_cfg=None,
                    **kwargs):
    """Forward function for training mode.
    Returns:
        dict[str, Tensor]: A dictionary of loss components.
    """
    assert proposal_cfg is None, '"proposal_cfg" must be None'
    # 前向推理结果，后面有分析
    outs = self(x, img_metas)
    if gt_labels is None:
        loss_inputs = outs + (gt_bboxes, img_metas)
    else:
        loss_inputs = outs + (gt_bboxes, gt_labels, img_metas)
    losses = self.loss(*loss_inputs, gt_bboxes_ignore=gt_bboxes_ignore)
    return losses
```



### forward&forward_single



```python
def forward(self, feats, img_metas):
    """Forward function.

    Args:
        feats (tuple[Tensor]): Features from the upstream network, each is
            a 4D-tensor.
        img_metas (list[dict]): List of image information.

    Returns:
        tuple[list[Tensor], list[Tensor]]: Outputs for all scale levels.

            - all_cls_scores_list (list[Tensor]): Classification scores \
                for each scale level. Each is a 4D-tensor with shape \
                [nb_dec, bs, num_query, cls_out_channels]. Note \
                `cls_out_channels` should includes background.
            - all_bbox_preds_list (list[Tensor]): Sigmoid regression \
                outputs for each scale level. Each is a 4D-tensor with \
                normalized coordinate format (cx, cy, w, h) and shape \
                [nb_dec, bs, num_query, 4].
    """
    # 这里是 1， 因为 DETR 默认用最后一层特征图
    num_levels = len(feats)
    img_metas_list = [img_metas for _ in range(num_levels)]
    return multi_apply(self.forward_single, feats, img_metas_list)
```

直接看 forward_single，里面是 head 前向的逻辑。

```python
def forward_single(self, x, img_metas):
    """"Forward function for a single feature level.

    Args:
        x (Tensor): Input feature from backbone's single stage, shape
            [bs, c, h, w].
        img_metas (list[dict]): List of image information.

    Returns:
        all_cls_scores (Tensor): Outputs from the classification head,
            shape [nb_dec, bs, num_query, cls_out_channels]. Note
            cls_out_channels should includes background.
        all_bbox_preds (Tensor): Sigmoid outputs from the regression
            head with normalized coordinate format (cx, cy, w, h).
            Shape [nb_dec, bs, num_query, 4].
    """
    # construct binary masks which used for the transformer.
    # NOTE following the official DETR repo, non-zero values representing
    # ignored positions, while zero values means valid positions.
    batch_size = x.size(0)
    # batch 中每张图的 batch_input_shape 都是一样的
    input_img_h, input_img_w = img_metas[0]['batch_input_shape']
    # 先将 mask 设置为全 1
    masks = x.new_ones((batch_size, input_img_h, input_img_w))
    # 对每一张图来说，在原来图片有像素的地方把 mask 置 0
    # 因此 mask 中 padding 的地方才是 1
    for img_id in range(batch_size):
        img_h, img_w, _ = img_metas[img_id]['img_shape']
        masks[img_id, :img_h, :img_w] = 0

    # 将每一层的特征图先投影到指定的特征维度
    # self.input_proj = Conv2d(self.in_channels, self.embed_dims, kernel_size=1)
    x = self.input_proj(x)	# shape：（B，embed_dims，H，W）
    # interpolate masks to have the same spatial shape with x
    # masks: [B, H, W]
    masks = F.interpolate(
        masks.unsqueeze(1), size=x.shape[-2:]).to(torch.bool).squeeze(1)
    # position encoding
    # 得到位置编码 shape：[B, 256, H, W]
    pos_embed = self.positional_encoding(masks)  # [bs, embed_dim, h, w]
    # outs_dec: [nb_dec, bs, num_query, embed_dim]
    outs_dec, _ = self.transformer(x, masks, self.query_embedding.weight,
                                    pos_embed)
		# 对 query 进行分类和回归
    # shape [num_decoder, B, num_query, num_class+1]
    all_cls_scores = self.fc_cls(outs_dec)
    # 经过 ffn 再经过一个卷积得到 4 个输出的值，经过 sigmoid 归一化到 0-1，输出的是 xyhw
    # shape [num_decoder, B, num_query, 4]
    all_bbox_preds = self.fc_reg(self.activate(
        self.reg_ffn(outs_dec))).sigmoid()
    return all_cls_scores, all_bbox_preds
```

### loss



来这里看看 DETR 怎么计算 loss 的			

```python
@force_fp32(apply_to=('all_cls_scores_list', 'all_bbox_preds_list'))
def loss(self,
            all_cls_scores_list,
            all_bbox_preds_list,
            gt_bboxes_list,
            gt_labels_list,
            img_metas,
            gt_bboxes_ignore=None):
    """"Loss function.

    Only outputs from the last feature level are used for computing
    losses by default.

    Args:
        all_cls_scores_list (list[Tensor]): Classification outputs
            for each feature level. Each is a 4D-tensor with shape
            [nb_dec, bs, num_query, cls_out_channels].
        all_bbox_preds_list (list[Tensor]): Sigmoid regression
            outputs for each feature level. Each is a 4D-tensor with
            normalized coordinate format (cx, cy, w, h) and shape
            [nb_dec, bs, num_query, 4].
        gt_bboxes_list (list[Tensor]): Ground truth bboxes for each image
            with shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
        gt_labels_list (list[Tensor]): Ground truth class indices for each
            image with shape (num_gts, ).
        img_metas (list[dict]): List of image meta information.
        gt_bboxes_ignore (list[Tensor], optional): Bounding boxes
            which can be ignored for each image. Default None.

    Returns:
        dict[str, Tensor]: A dictionary of loss components.
    """
    # NOTE defaultly only the outputs from the last feature scale is used.
    # shape:  [num_decoder, B, num_query, num_class+1]
    all_cls_scores = all_cls_scores_list[-1]
    # shape:  [num_decoder, B, num_query, 4]
    all_bbox_preds = all_bbox_preds_list[-1]
    assert gt_bboxes_ignore is None, \
        'Only supports for gt_bboxes_ignore setting to None.'
		
    # decoder 的层数，默认是 6
    num_dec_layers = len(all_cls_scores)
    all_gt_bboxes_list = [gt_bboxes_list for _ in range(num_dec_layers)]
    all_gt_labels_list = [gt_labels_list for _ in range(num_dec_layers)]
    all_gt_bboxes_ignore_list = [
        gt_bboxes_ignore for _ in range(num_dec_layers)
    ]
    img_metas_list = [img_metas for _ in range(num_dec_layers)]
		
    # 调用 loss_single 函数
    losses_cls, losses_bbox, losses_iou = multi_apply(
        self.loss_single, all_cls_scores, all_bbox_preds,
        all_gt_bboxes_list, all_gt_labels_list, img_metas_list,
        all_gt_bboxes_ignore_list)

    # 分别计算每一层 decoder 的 loss
    loss_dict = dict()
    # loss from the last decoder layer
    loss_dict['loss_cls'] = losses_cls[-1]
    loss_dict['loss_bbox'] = losses_bbox[-1]
    loss_dict['loss_iou'] = losses_iou[-1]
    # loss from other decoder layers
    num_dec_layer = 0
    for loss_cls_i, loss_bbox_i, loss_iou_i in zip(losses_cls[:-1],
                                                    losses_bbox[:-1],
                                                    losses_iou[:-1]):
        loss_dict[f'd{num_dec_layer}.loss_cls'] = loss_cls_i
        loss_dict[f'd{num_dec_layer}.loss_bbox'] = loss_bbox_i
        loss_dict[f'd{num_dec_layer}.loss_iou'] = loss_iou_i
        num_dec_layer += 1
    return loss_dict
```



### loss_single



主要的 loss 逻辑在这里

```python
def loss_single(self,
                cls_scores,
                bbox_preds,
                gt_bboxes_list,
                gt_labels_list,
                img_metas,
                gt_bboxes_ignore_list=None):
    """"Loss function for outputs from a single decoder layer of a single
    feature level.

    Args:
        cls_scores (Tensor): Box score logits from a single decoder layer
            for all images. Shape [bs, num_query, cls_out_channels].
        bbox_preds (Tensor): Sigmoid outputs from a single decoder layer
            for all images, with normalized coordinate (cx, cy, w, h) and
            shape [bs, num_query, 4].
        gt_bboxes_list (list[Tensor]): Ground truth bboxes for each image
            with shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
        gt_labels_list (list[Tensor]): Ground truth class indices for each
            image with shape (num_gts, ).
        img_metas (list[dict]): List of image meta information.
        gt_bboxes_ignore_list (list[Tensor], optional): Bounding
            boxes which can be ignored for each image. Default None.

    Returns:
        dict[str, Tensor]: A dictionary of loss components for outputs from
            a single decoder layer.
    """
    num_imgs = cls_scores.size(0)
    cls_scores_list = [cls_scores[i] for i in range(num_imgs)]
    bbox_preds_list = [bbox_preds[i] for i in range(num_imgs)]
    cls_reg_targets = self.get_targets(cls_scores_list, bbox_preds_list,
                                        gt_bboxes_list, gt_labels_list,
                                        img_metas, gt_bboxes_ignore_list)
    (labels_list, label_weights_list, bbox_targets_list, bbox_weights_list,
        num_total_pos, num_total_neg) = cls_reg_targets
    labels = torch.cat(labels_list, 0)
    label_weights = torch.cat(label_weights_list, 0)
    bbox_targets = torch.cat(bbox_targets_list, 0)
    bbox_weights = torch.cat(bbox_weights_list, 0)

    # classification loss
    cls_scores = cls_scores.reshape(-1, self.cls_out_channels)
    # construct weighted avg_factor to match with the official DETR repo
    cls_avg_factor = num_total_pos * 1.0 + \
        num_total_neg * self.bg_cls_weight
    loss_cls = self.loss_cls(
        cls_scores, labels, label_weights, avg_factor=cls_avg_factor)

    # Compute the average number of gt boxes accross all gpus, for
    # normalization purposes
    num_total_pos = loss_cls.new_tensor([num_total_pos])
    num_total_pos = torch.clamp(reduce_mean(num_total_pos), min=1).item()

    # construct factors used for rescale bboxes
    factors = []
    for img_meta, bbox_pred in zip(img_metas, bbox_preds):
        img_h, img_w, _ = img_meta['img_shape']
        factor = bbox_pred.new_tensor([img_w, img_h, img_w,
                                        img_h]).unsqueeze(0).repeat(
                                            bbox_pred.size(0), 1)
        factors.append(factor)
    factors = torch.cat(factors, 0)

    # DETR regress the relative position of boxes (cxcywh) in the image,
    # thus the learning target is normalized by the image size. So here
    # we need to re-scale them for calculating IoU loss
    bbox_preds = bbox_preds.reshape(-1, 4)
    bboxes = bbox_cxcywh_to_xyxy(bbox_preds) * factors
    bboxes_gt = bbox_cxcywh_to_xyxy(bbox_targets) * factors

    # regression IoU loss, defaultly GIoU loss
    loss_iou = self.loss_iou(
        bboxes, bboxes_gt, bbox_weights, avg_factor=num_total_pos)

    # regression L1 loss
    loss_bbox = self.loss_bbox(
        bbox_preds, bbox_targets, bbox_weights, avg_factor=num_total_pos)
    return loss_cls, loss_bbox, loss_iou

```









## SinePositionalEncoding



这个类对特征图中有像素的地方生成位置编码，以减缓 transformer 丢失位置信息，在 config 中默认使用 sin 形式的位置编码，并且防止编码太大，normalize 到 0-1 之间

```python
positional_encoding=dict(
  type='SinePositionalEncoding', num_feats=128, normalize=True),
```

见注释 (感觉这里理解的不是很深刻)

```python
@POSITIONAL_ENCODING.register_module()
class SinePositionalEncoding(nn.Module):
    """Position encoding with sine and cosine functions.

    See `End-to-End Object Detection with Transformers
    <https://arxiv.org/pdf/2005.12872>`_ for details.

    Args:
        num_feats (int): The feature dimension for each position
            along x-axis or y-axis. Note the final returned dimension
            for each position is 2 times of this value.
        temperature (int, optional): The temperature used for scaling
            the position embedding. Default 10000.
        normalize (bool, optional): Whether to normalize the position
            embedding. Default False.
        scale (float, optional): A scale factor that scales the position
            embedding. The scale will be used only when `normalize` is True.
            Default 2*pi.
        eps (float, optional): A value added to the denominator for
            numerical stability. Default 1e-6.
    """

    def __init__(self,
                 num_feats,
                 temperature=10000,
                 normalize=False,
                 scale=2 * math.pi,
                 eps=1e-6):
        super(SinePositionalEncoding, self).__init__()
        if normalize:
            assert isinstance(scale, (float, int)), 'when normalize is set,' \
                'scale should be provided and in float or int type, ' \
                f'found {type(scale)}'
        self.num_feats = num_feats
        self.temperature = temperature
        self.normalize = normalize
        self.scale = scale
        self.eps = eps

    def forward(self, mask):
        """Forward function for `SinePositionalEncoding`.

        Args:
            mask (Tensor): ByteTensor mask. Non-zero values representing
                ignored positions, while zero values means valid positions
                for this image. Shape [bs, h, w].

        Returns:
            pos (Tensor): Returned position embedding with shape
                [bs, num_feats*2, h, w].
        """
        # not_mask 就是表示有像素的地方就是 1，没有的地方就是 0
        # shape：[B, H, W]
        not_mask = ~mask
        # y 方向累加，(1，1，1)->(1，2，3)
        # (1,1,1,...) #y_embed
        # (2,2,2,...)
        # (3,3,3,...)
        # (...)
        y_embed = not_mask.cumsum(1, dtype=torch.float32)
        # x 方向累加，(1，1，1)->(1，2，3)
        # (1,2,3,...) #x_embed
        # (1,2,3,...)
        # (1,2,3,...)
        # (...)
        x_embed = not_mask.cumsum(2, dtype=torch.float32)
        if self.normalize:
          	# 将编码归一化，y_embed[:, -1:, :] -> shape [B, 1, W]
            # 保留了 y 方向上最大的编码数，防止除以 0 加上了 eps
            # 然后乘上了scale，默认是 2pi，所以最终结果为 0-2pi 之间
            # 列表 l[-1] 和 l[-1:] 是不一样的，前者返回一个值，后者返回只有一个值的列表
            y_embed = y_embed / (y_embed[:, -1:, :] + self.eps) * self.scale
            x_embed = x_embed / (x_embed[:, :, -1:] + self.eps) * self.scale
        # dim_t -> [0,...,127]，
        # 这个数是 FPN 通道数的一半，因为最终要 concat 在一起然后和特征图相加起来
        dim_t = torch.arange(
            self.num_feats, dtype=torch.float32, device=mask.device)
        # 按照公式进行
        # (2 * (dim_t // 2) -> [0,0,2,2,...,126,126]
        # 每一个通道上的 sin 函数的周期都不一样，dim 越大周期越大，这里算出来相当于 sin(wx) 函数中的 w
        dim_t = self.temperature**(2 * (dim_t // 2) / self.num_feats)
        # shape: [B, H, W, 128]
        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        # 算出每一个 position 在 x 和 y 方向上的位置编码
        # shape: [B, H, W, 128]
        pos_x = torch.stack(
            (pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()),
            dim=4).flatten(3)
        pos_y = torch.stack(
            (pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()),
            dim=4).flatten(3)
        # shape: [B, H, W, 256] -> [B, 256, H, W]
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
        return pos
```

> 一些比较好理解的 Positional Encoding 的博客
>
> https://zhuanlan.zhihu.com/p/166244505
>
> https://blog.csdn.net/weixin_42715977/article/details/122135262

## FFN



在 TransformerHead 中解码出向量之后，经过 FFN 得到每一个 query 最终的预测结果, DETR 中 FFN 的实例化如下

```python
self.reg_ffn = FFN(
    self.embed_dims,	# 256, 和 FPN 通道一样
    self.embed_dims,
    self.num_fcs,	# 2
    self.act_cfg, # ReLU
    dropout=0.0,
    add_residual=False)
```

FFN 代码如下，就是进行两层 FC，然后输出

```python
class FFN(nn.Module):
    """Implements feed-forward networks (FFNs) with residual connection.

    Args:
        embed_dims (int): The feature dimension. Same as
            `MultiheadAttention`.
        feedforward_channels (int): The hidden dimension of FFNs.
        num_fcs (int, optional): The number of fully-connected layers in
            FFNs. Defaults to 2.
        act_cfg (dict, optional): The activation config for FFNs.
        dropout (float, optional): Probability of an element to be
            zeroed. Default 0.0.
        add_residual (bool, optional): Add resudual connection.
            Defaults to True.
    """

    def __init__(self,
                 embed_dims,
                 feedforward_channels,
                 num_fcs=2,
                 act_cfg=dict(type='ReLU', inplace=True),
                 dropout=0.0,
                 add_residual=True):
        super(FFN, self).__init__()
        assert num_fcs >= 2, 'num_fcs should be no less ' \
            f'than 2. got {num_fcs}.'
        self.embed_dims = embed_dims
        self.feedforward_channels = feedforward_channels
        self.num_fcs = num_fcs
        self.act_cfg = act_cfg
        self.dropout = dropout
        self.activate = build_activation_layer(act_cfg)

        layers = nn.ModuleList()
        in_channels = embed_dims
        for _ in range(num_fcs - 1):
            layers.append(
                nn.Sequential(
                    Linear(in_channels, feedforward_channels), self.activate,
                    nn.Dropout(dropout)))
            in_channels = feedforward_channels
        layers.append(Linear(feedforward_channels, embed_dims))
        self.layers = nn.Sequential(*layers)
        self.dropout = nn.Dropout(dropout)
        self.add_residual = add_residual

    def forward(self, x, residual=None):
        """Forward function for `FFN`."""
        out = self.layers(x)
        if not self.add_residual:
            return out
        if residual is None:
            residual = x
        return residual + self.dropout(out)

    def __repr__(self):
        """str: a string that describes the module"""
        repr_str = self.__class__.__name__
        repr_str += f'(embed_dims={self.embed_dims}, '
        repr_str += f'feedforward_channels={self.feedforward_channels}, '
        repr_str += f'num_fcs={self.num_fcs}, '
        repr_str += f'act_cfg={self.act_cfg}, '
        repr_str += f'dropout={self.dropout}, '
        repr_str += f'add_residual={self.add_residual})'
        return repr_str
```





## HungarianAssigner



```python
import torch

from ..builder import BBOX_ASSIGNERS
from ..match_costs import build_match_cost
from ..transforms import bbox_cxcywh_to_xyxy
from .assign_result import AssignResult
from .base_assigner import BaseAssigner

try:
    from scipy.optimize import linear_sum_assignment
except ImportError:
    linear_sum_assignment = None


@BBOX_ASSIGNERS.register_module()
class HungarianAssigner(BaseAssigner):
    """Computes one-to-one matching between predictions and ground truth.

    This class computes an assignment between the targets and the predictions
    based on the costs. The costs are weighted sum of three components:
    classification cost, regression L1 cost and regression iou cost. The
    targets don't include the no_object, so generally there are more
    predictions than targets. After the one-to-one matching, the un-matched
    are treated as backgrounds. Thus each query prediction will be assigned
    with `0` or a positive integer indicating the ground truth index:

    - 0: negative sample, no assigned gt
    - positive integer: positive sample, index (1-based) of assigned gt

    Args:
        cls_weight (int | float, optional): The scale factor for classification
            cost. Default 1.0.
        bbox_weight (int | float, optional): The scale factor for regression
            L1 cost. Default 1.0.
        iou_weight (int | float, optional): The scale factor for regression
            iou cost. Default 1.0.
        iou_calculator (dict | optional): The config for the iou calculation.
            Default type `BboxOverlaps2D`.
        iou_mode (str | optional): "iou" (intersection over union), "iof"
                (intersection over foreground), or "giou" (generalized
                intersection over union). Default "giou".
    """

    def __init__(self,
                 cls_cost=dict(type='ClassificationCost', weight=1.),
                 reg_cost=dict(type='BBoxL1Cost', weight=1.0),
                 iou_cost=dict(type='IoUCost', iou_mode='giou', weight=1.0)):
        self.cls_cost = build_match_cost(cls_cost)
        self.reg_cost = build_match_cost(reg_cost)
        self.iou_cost = build_match_cost(iou_cost)

    def assign(self,
               bbox_pred,
               cls_pred,
               gt_bboxes,
               gt_labels,
               img_meta,
               gt_bboxes_ignore=None,
               eps=1e-7):
        """Computes one-to-one matching based on the weighted costs.

        This method assign each query prediction to a ground truth or
        background. The `assigned_gt_inds` with -1 means don't care,
        0 means negative sample, and positive number is the index (1-based)
        of assigned gt.
        The assignment is done in the following steps, the order matters.

        1. assign every prediction to -1
        2. compute the weighted costs
        3. do Hungarian matching on CPU based on the costs
        4. assign all to 0 (background) first, then for each matched pair
           between predictions and gts, treat this prediction as foreground
           and assign the corresponding gt index (plus 1) to it.

        Args:
            bbox_pred (Tensor): Predicted boxes with normalized coordinates
                (cx, cy, w, h), which are all in range [0, 1]. Shape
                [num_query, 4].
            cls_pred (Tensor): Predicted classification logits, shape
                [num_query, num_class].
            gt_bboxes (Tensor): Ground truth boxes with unnormalized
                coordinates (x1, y1, x2, y2). Shape [num_gt, 4].
            gt_labels (Tensor): Label of `gt_bboxes`, shape (num_gt,).
            img_meta (dict): Meta information for current image.
            gt_bboxes_ignore (Tensor, optional): Ground truth bboxes that are
                labelled as `ignored`. Default None.
            eps (int | float, optional): A value added to the denominator for
                numerical stability. Default 1e-7.

        Returns:
            :obj:`AssignResult`: The assigned result.
        """
        assert gt_bboxes_ignore is None, \
            'Only case when gt_bboxes_ignore is None is supported.'
        num_gts, num_bboxes = gt_bboxes.size(0), bbox_pred.size(0)

        # 1. assign -1 by default
        # shape： num_query
        assigned_gt_inds = bbox_pred.new_full((num_bboxes, ),
                                              -1,
                                              dtype=torch.long)
        assigned_labels = bbox_pred.new_full((num_bboxes, ),
                                             -1,
                                             dtype=torch.long)
        if num_gts == 0 or num_bboxes == 0:
            # No ground truth or boxes, return empty assignment
            if num_gts == 0:
                # No ground truth, assign all to background
                assigned_gt_inds[:] = 0
            return AssignResult(
                num_gts, assigned_gt_inds, None, labels=assigned_labels)
        img_h, img_w, _ = img_meta['img_shape']
        factor = gt_bboxes.new_tensor([img_w, img_h, img_w,
                                       img_h]).unsqueeze(0)

        # 2. compute the weighted costs
        # classification and bboxcost.
        cls_cost = self.cls_cost(cls_pred, gt_labels)
        # regression L1 cost
        # L1 cost 需要归一化坐标
        normalize_gt_bboxes = gt_bboxes / factor
        reg_cost = self.reg_cost(bbox_pred, normalize_gt_bboxes)
        # regression iou cost, defaultly giou is used in official DETR.
        # IoU cost 用 GIoU 来衡量，需要 xyxy 形式的绝对坐标
        bboxes = bbox_cxcywh_to_xyxy(bbox_pred) * factor
        iou_cost = self.iou_cost(bboxes, gt_bboxes)
        # weighted sum of above three costs
        # shape: [num_query, num_gt]
        cost = cls_cost + reg_cost + iou_cost

        # 3. do Hungarian matching on CPU using linear_sum_assignment
        cost = cost.detach().cpu()
        if linear_sum_assignment is None:
            raise ImportError('Please run "pip install scipy" '
                              'to install scipy first.')
        # 根据 cost，利用匈牙利算法对每个 gt 匹配一个 query，使得被匹配的 query 的总 cost 最小
        # 下一小节的 demo 有介绍
        matched_row_inds, matched_col_inds = linear_sum_assignment(cost)
        # shape： num_gt
        matched_row_inds = torch.from_numpy(matched_row_inds).to(
            bbox_pred.device)
        # shape： num_gt
        matched_col_inds = torch.from_numpy(matched_col_inds).to(
            bbox_pred.device)

        # 4. assign backgrounds and foregrounds
        # assign all indices to backgrounds first
        # 先给每一个 query 都匹配到背景类
        assigned_gt_inds[:] = 0
        # assign foregrounds based on matching results
        # 有匹配的 query 匹配的 gt 的索引数 (从 1 开始)
        assigned_gt_inds[matched_row_inds] = matched_col_inds + 1
        # 有匹配的 query 负责分类的标签
        assigned_labels[matched_row_inds] = gt_labels[matched_col_inds]
        # 没有被匹配的 query 就被认为是背景
        return AssignResult(
            num_gts, assigned_gt_inds, None, labels=assigned_labels)

```



### demo



假设下面 row 是 num_query，col 是 num_gt，那么我们要将每一个 gt 只匹配给一个 query，匹配原则是让他们的总 cost 最小，那么 2 1 2 是最优选择，坐标就是 [0,1] [1,0] [2,2]，row_ind=[0,1,2]，col_ind=[1,0,2]

```python
cost = np.array([[4, 1, 3], 
                 [2, 0, 5], 
                 [3, 2, 2]])
from scipy.optimize import linear_sum_assignment
row_ind, col_ind = linear_sum_assignment(cost)
col_ind
array([1, 0, 2])
cost[row_ind, col_ind].sum()
5
```





## CostFunction



detr 做二分匹配时根据 cost function 的大小来为每一个 prediction 分配标签，主要用到了三个 cost function，都在 `mmdet/core/bbox/match_costs.py` 里面 ，下面介绍。



### ClassificationCost



见注释

```python
MATCH_COST.register_module()
class ClassificationCost(object):
    """ClsSoftmaxCost.

     Args:
         weight (int | float, optional): loss_weight

     Examples:
         >>> from mmdet.core.bbox.match_costs.match_cost import \
         ... ClassificationCost
         >>> import torch
         >>> self = ClassificationCost()
         >>> cls_pred = torch.rand(4, 3)
         >>> gt_labels = torch.tensor([0, 1, 2])
         >>> factor = torch.tensor([10, 8, 10, 8])
         >>> self(cls_pred, gt_labels)
         tensor([[-0.3430, -0.3525, -0.3045],
                [-0.3077, -0.2931, -0.3992],
                [-0.3664, -0.3455, -0.2881],
                [-0.3343, -0.2701, -0.3956]])
    """

    def __init__(self, weight=1.):
        self.weight = weight

    def __call__(self, cls_pred, gt_labels):
        """
        Args:
            cls_pred (Tensor): Predicted classification logits, shape
                [num_query, num_class].
            gt_labels (Tensor): Label of `gt_bboxes`, shape (num_gt,).

        Returns:
            torch.Tensor: cls_cost value with weight
        """
        # Following the official DETR repo, contrary to the loss that
        # NLL is used, we approximate it in 1 - cls_score[gt_label].
        # The 1 is a constant that doesn't change the matching,
        # so it can be omitted.
        # shape: [num_query, num_class]
        cls_score = cls_pred.softmax(-1)
        # shape: [num_query, num_gt]
        # 返回每一个 prediction 对每一个 gt_label 的 cost，越小代表得分越高
        cls_cost = -cls_score[:, gt_labels]
        return cls_cost * self.weight

```



### IoUCost



见注释

```python
@MATCH_COST.register_module()
class IoUCost(object):
    """IoUCost.

     Args:
         iou_mode (str, optional): iou mode such as 'iou' | 'giou'
         weight (int | float, optional): loss weight

     Examples:
         >>> from mmdet.core.bbox.match_costs.match_cost import IoUCost
         >>> import torch
         >>> self = IoUCost()
         >>> bboxes = torch.FloatTensor([[1,1, 2, 2], [2, 2, 3, 4]])
         >>> gt_bboxes = torch.FloatTensor([[0, 0, 2, 4], [1, 2, 3, 4]])
         >>> self(bboxes, gt_bboxes)
         tensor([[-0.1250,  0.1667],
                [ 0.1667, -0.5000]])
    """

    def __init__(self, iou_mode='giou', weight=1.):
        self.weight = weight
        self.iou_mode = iou_mode

    def __call__(self, bboxes, gt_bboxes):
        """
        Args:
            bboxes (Tensor): Predicted boxes with unnormalized coordinates
                (x1, y1, x2, y2). Shape [num_query, 4].
            gt_bboxes (Tensor): Ground truth boxes with unnormalized
                coordinates (x1, y1, x2, y2). Shape [num_gt, 4].

        Returns:
            torch.Tensor: iou_cost value with weight
        """
        # overlaps: [num_query, num_gt]
        # 返回一一配对的 IoU
        overlaps = bbox_overlaps(
            bboxes, gt_bboxes, mode=self.iou_mode, is_aligned=False)
        # The 1 is a constant that doesn't change the matching, so omitted.
        # IoU 越大，cost 越小
        iou_cost = -overlaps
        return iou_cost * self.weight

```



### BBoxL1Cost



见注释

```python
@MATCH_COST.register_module()
class BBoxL1Cost(object):
    """BBoxL1Cost.

     Args:
         weight (int | float, optional): loss_weight
         box_format (str, optional): 'xyxy' for DETR, 'xywh' for Sparse_RCNN

     Examples:
         >>> from mmdet.core.bbox.match_costs.match_cost import BBoxL1Cost
         >>> import torch
         >>> self = BBoxL1Cost()
         >>> bbox_pred = torch.rand(1, 4)
         >>> gt_bboxes= torch.FloatTensor([[0, 0, 2, 4], [1, 2, 3, 4]])
         >>> factor = torch.tensor([10, 8, 10, 8])
         >>> self(bbox_pred, gt_bboxes, factor)
         tensor([[1.6172, 1.6422]])
    """

    def __init__(self, weight=1., box_format='xyxy'):
        self.weight = weight
        assert box_format in ['xyxy', 'xywh']
        self.box_format = box_format

    def __call__(self, bbox_pred, gt_bboxes):
        """
        Args:
            bbox_pred (Tensor): Predicted boxes with normalized coordinates
                (cx, cy, w, h), which are all in range [0, 1]. Shape
                [num_query, 4].
            gt_bboxes (Tensor): Ground truth boxes with normalized
                coordinates (x1, y1, x2, y2). Shape [num_gt, 4].

        Returns:
            torch.Tensor: bbox_cost value with weight
        """
        # 注意这个是经过缩放的坐标，是 0-1 范围的
        if self.box_format == 'xywh':
            gt_bboxes = bbox_xyxy_to_cxcywh(gt_bboxes)
        elif self.box_format == 'xyxy':
            bbox_pred = bbox_cxcywh_to_xyxy(bbox_pred)
        # 每一个 prediction box 到 gt_box 的距离，越大说明 cost 越大
        # shape: [num_query, num_gt]
        bbox_cost = torch.cdist(bbox_pred, gt_bboxes, p=1)
        return bbox_cost * self.weight
```



## reference



https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.linear_sum_assignment.html#scipy.optimize.linear_sum_assignment

https://zhuanlan.zhihu.com/p/348060767

https://zhuanlan.zhihu.com/p/345985277



我的博客即将同步至腾讯云开发者社区，邀请大家一同入驻：https://cloud.tencent.com/developer/support-plan?invite_code=38seq1wzumm8k
