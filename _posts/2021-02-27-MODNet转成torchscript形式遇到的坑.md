---
layout: post
title: MODNet转成torchscript形式遇到的坑
subtitle: 
date: 2021-02-27
author: kevin
header-img: img/green-bg.jpg
catalog: true
tags:
    - deep learning
    - python
    - image matting
---



## preface



项目需要，就将 [MODNet](https://github.com/ZHKKKe/MODNet) 官方提供的模型转成 torchscript 形式，方便在手机上也能够调用



## torch.jit.script 不支持 lambda 表达式，所以要拆开来写模型前向代码



```
torch.jit.frontend.UnsupportedNodeError: Lambda aren't supported:
  File "/raid/kevin/github/image-matting/srcs/models/backbones/mobilenetv2.py", line 141
        def forward(self, x, feature_names=None):
                # Stage1
                x = reduce(lambda x, n: self.features[n](x), list(range(0,2)), x)
             ~ <--- HERE
                # Stage2
                x = reduce(lambda x, n: self.features[n](x), list(range(2,4)), x)
```



因此，我将他们拆开成如下形式

```python
x = self.model.features[0](x)
x = self.model.features[1](x)
x = self.model.features[2](x)
x = self.model.features[3](x)
x = self.model.features[4](x)
x = self.model.features[5](x)
x = self.model.features[6](x)
……
```



## 默认值为 None 会报错，要么删掉要么赋值其他的东西



```
RuntimeError: 
Expected a default value of type Tensor (inferred) on parameter "feature_names".Because "feature_names" was not annotated with an explicit type it is assumed to be type 'Tensor'.:
  File "/raid/kevin/github/image-matting/srcs/models/backbones/mobilenetv2.py", line 139
        def forward(self, x, feature_names=None):
 ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~...  <--- HERE
                # Stage1
                x = self.features[0](x)

```

 把里面的 feature_names 去掉就行了，在代码里面也没有用到



## F.interpolate 里面的 scale_factor 要求浮点数，不能是 int



```
RuntimeError: 
Arguments for call are not valid.
The following variants are available:

  interpolate(Tensor input, int[]? size=None, float? scale_factor=None, str mode="nearest", bool? align_corners=None, bool? recompute_scale_factor=None) -> (Tensor):
  Expected a value of type 'Optional[float]' for argument 'scale_factor' but instead found type 'int'.

The original call is:
  File "/raid/kevin/github/image-matting/srcs/models/modnet.py", line 110
        enc32x = self.se_block(enc32x)
        # 再上采样4倍
        lr16x = F.interpolate(enc32x, scale_factor=2, mode='bilinear', align_corners=False)
                ~~~~~~~~~~~~~ <--- HERE
        lr16x = self.conv_lr16x(lr16x)
        lr8x = F.interpolate(lr16x, scale_factor=2, mode='bilinear', align_corners=False)
```



解决办法就是将 scale_factor=2 变成 2.0



## 先给 predict_details 赋值为  None 再赋值为一个 tensor 时会报错，需要初始化其为一个 tensor 类型数据



```
RuntimeError: 
Variable 'pred_semantic' previously has type None but is now being assigned to a value of type Tensor
:
  File "/raid/kevin/github/image-matting/srcs/models/modnet.py", line 118
        if not inference:
            lr = self.conv_lr(lr8x)
            pred_semantic = torch.sigmoid(lr)
            ~~~~~~~~~~~~~ <--- HERE

        return pred_semantic, lr8x, [enc2x, enc4x] 
```



我将 `pred_semantic = None` 变成了 `pred_semantic = torch.tensor([])`



## jit 不支持 data_parallel



```
torch.jit.frontend.NotSupportedError: Compiled functions can't take variable number of arguments or use keyword-only arguments with defaults:
  File "/raid/kevin/anaconda3/envs/open-mmlab/lib/python3.7/site-packages/torch/nn/parallel/data_parallel.py", line 141
    def forward(self, *inputs, **kwargs):
                                ~~~~~~~ <--- HERE
        if not self.device_ids:
            return self.module(*inputs, **kwargs)

```



网上没有相关解决方案，那没办法，就不用 `nn.DataParallel()` ，作者提供的预训练模型是在多卡下训练的，所以模型参数的 key 前面会有一个 `module.xx` 前缀，比如正常的参数名是 `hr_branch.conv_hr.1.layers.0.weight`，如果是多卡的话名称就会变成 `module.hr_branch.conv_hr.1.layers.0.weight`，因此我们只需要将 key 的前缀去掉，然后让模型加载新的参数就行了



```python
# create MODNet and load the pre-trained ckpt
modnet = MODNet(backbone_pretrained=True)
# modnet = nn.DataParallel(modnet).cuda()
modnet = modnet.cuda()
ckpt = torch.load(args.ckpt, map_location='cpu')

# if use more than one GPU
ckpt_single = OrderedDict()
for k, v in ckpt.items():
    k = k.replace('module.', '')
    ckpt_single[k] = v

modnet.load_state_dict(ckpt_single)
modnet.eval()
```

