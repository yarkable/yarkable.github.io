---
layout:     post
title:      MMDetection & pycocotools eval 详解
subtitle:   
date:       2023-03-20
author:     Kevin
header-img: img/green-bg.jpg
catalog: true
tags:
    - linux
    - object detection
    - deep learning
    - mmdetection

---

#  preface

记录 mmdet 对检测器进行评估的过程，以 COCO 数据集为例，所使用到的 mmdet 版本为 2.18.0。本质上其实就是对 pycocotools 的封装调用，特此记录，方便复习。

# test.py

首先是在 `tools/test.py` 产生推理过后的结果，然后再用每一个数据集的 `evaluate` 函数进行性能的评估

```Python
if not distributed:
    model = MMDataParallel(model, device_ids=[0])
    # 调用检测器的前向推理函数得到推理的结果
    outputs = single_gpu_test(model, data_loader, args.show, args.show_dir,
                              args.show_score_thr)
else:
    model = MMDistributedDataParallel(
        model.cuda(),
        device_ids=[torch.cuda.current_device()],
        broadcast_buffers=False)
    outputs = multi_gpu_test(model, data_loader, args.tmpdir,
                             args.gpu_collect)

# 上面得到的 outputs 是一个列表
# 列表长度为验证集的图片数
# 里面每一个元素的长度都是数据集的类别数
# 再里面的每一个元素就是检测器检测出来的对应该类别的框坐标和置信度（经过阈值过滤和 nms 了）
# 拿 COCO 来说，这里的列表长度就是 5000，里面每一个元素又是长度为 80 的列表
# 里面的结果已经经过阈值过滤以及利用 iou 阈值进行 nms，相关参数在 Head 的 test_cfg，每个类别保留的框的最大个数默认为 100
rank, _ = get_dist_info()
if rank == 0:
    if args.out:
        print(f'\nwriting results to {args.out}')
        mmcv.dump(outputs, args.out)
    kwargs = {} if args.eval_options is None else args.eval_options
    # 这个选项调用 _det2json 将大列表转成 COCO 标准的 json 格式的列表，一般用于提交检测结果至服务器评估
    # {'image_id': x, 'bbox': x, 'score': x, 'category_id': x}
    if args.format_only:
        dataset.format_results(outputs, **kwargs)
    # 这里是 eval 的逻辑
    if args.eval:
        # 如果是 coco 的话一般这里的评估方式是 bbox
        eval_kwargs = cfg.get('evaluation', {}).copy()
        # hard-code way to remove EvalHook args
        for key in [
                'interval', 'tmpdir', 'start', 'gpu_collect', 'save_best',
                'rule'
        ]:
            eval_kwargs.pop(key, None)
        eval_kwargs.update(dict(metric=args.eval, **kwargs))
        # 直接调用数据集的 evaluate 函数进行评估
        # 所以 mmdet 每一个数据集都必须实现这个方法
        metric = dataset.evaluate(outputs, **eval_kwargs)
        print(metric)
        metric_dict = dict(config=args.config, metric=metric)
        if args.work_dir is not None and rank == 0:
            mmcv.dump(metric_dict, json_file)
```

# 快速debug的脚本

如果是用官方提供的 test 脚本的话，每一次 debug 都得重新推理一遍数据集的结果，很麻烦，我们可以用 `--out` 参数将推理的结果保存成 pkl 格式，然后用下面的脚本直接导入，速度很快。下面这个脚本其实就是对 test.py 的精简版，只留了数据集相关的配置，我叫它为 `naive_test.py`。

```Python
import pickle
import argparse
from mmcv import Config, DictAction
from mmdet.datasets import build_dataset

def parse_args():
    parser = argparse.ArgumentParser(
        description='MMDet test (and eval) a model')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('results', help='pkl format infer results path')
    parser.add_argument(
        '--eval',
        type=str,
        nargs='+',
        help='evaluation metrics, which depends on the dataset, e.g., "bbox",'
        ' "segm", "proposal" for COCO, and "mAP", "recall" for PASCAL VOC')
    parser.add_argument(
        '--eval-options',
        nargs='+',
        action=DictAction,
        help='custom options for evaluation, the key-value pair in xxx=yyy '
        'format will be kwargs for dataset.evaluate() function')
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    cfg = Config.fromfile(args.config)    
    # 这句不加的话会报错，train 的时候会过滤掉没有标注的图片，所以最终的图片数不对
    cfg.data.test.test_mode = True 
    dataset = build_dataset(cfg.data.test)
    with open(args.results, 'rb') as f:
        outputs = pickle.load(f)
    kwargs = {} if args.eval_options is None else args.eval_options
    if args.eval:
        eval_kwargs = cfg.get('evaluation', {}).copy()
        # hard-code way to remove EvalHook args
        for key in [
                'interval', 'tmpdir', 'start', 'gpu_collect', 'save_best',
                'rule'
        ]:
            eval_kwargs.pop(key, None)
        eval_kwargs.update(dict(metric=args.eval, **kwargs))
        metric = dataset.evaluate(outputs, **eval_kwargs)
        print(metric)

if __name__ == '__main__':
    main()
```

# mmdet/dataset/coco.py

## evaluate函数



上面看到流程已经进入了数据集的 `evaluate` 函数，这里我们就看细看一下 evaluate 的细节

```Python
def evaluate(self,
             results,
             metric='bbox',
             logger=None,
             jsonfile_prefix=None,
             classwise=False,
             proposal_nums=(100, 300, 1000),
             iou_thrs=None,
             metric_items=None):
    """Evaluation in COCO protocol.

    Args:
        results (list[list | tuple]): Testing results of the dataset.
        metric (str | list[str]): Metrics to be evaluated. Options are
            'bbox', 'segm', 'proposal', 'proposal_fast'.
        logger (logging.Logger | str | None): Logger used for printing
            related information during evaluation. Default: None.
        jsonfile_prefix (str | None): The prefix of json files. It includes
            the file path and the prefix of filename, e.g., "a/b/prefix".
            If not specified, a temp file will be created. Default: None.
        classwise (bool): Whether to evaluating the AP for each class.
        proposal_nums (Sequence[int]): Proposal number used for evaluating
            recalls, such as recall@100, recall@1000.
            Default: (100, 300, 1000).
        iou_thrs (Sequence[float], optional): IoU threshold used for
            evaluating recalls/mAPs. If set to a list, the average of all
            IoUs will also be computed. If not specified, [0.50, 0.55,
            0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95] will be used.
            Default: None.
        metric_items (list[str] | str, optional): Metric items that will
            be returned. If not specified, ``['AR@100', 'AR@300',
            'AR@1000', 'AR_s@1000', 'AR_m@1000', 'AR_l@1000' ]`` will be
            used when ``metric=='proposal'``, ``['mAP', 'mAP_50', 'mAP_75',
            'mAP_s', 'mAP_m', 'mAP_l']`` will be used when
            ``metric=='bbox' or metric=='segm'``.

    Returns:
        dict[str, float]: COCO style evaluation metric.
    """
    # 变成列表方便统一遍历（本身传进来也就是列表）
    metrics = metric if isinstance(metric, list) else [metric]
    # COCO 允许的几种 metric
    allowed_metrics = ['bbox', 'segm', 'proposal', 'proposal_fast']
    for metric in metrics:
        if metric not in allowed_metrics:
            raise KeyError(f'metric {metric} is not supported')
    if iou_thrs is None:
        # 一般 iou_thrs 不会特意设置，所以默认是计算 mAP 的
        # 如果只想计算某个 iou 下的 AP 可以在函数中传入这个参数
        iou_thrs = np.linspace(
            .5, 0.95, int(np.round((0.95 - .5) / .05)) + 1, endpoint=True)
    if metric_items is not None:
        # 这个一般也不会设置，所以默认返回所有的 metric，
        # 想单独返回某一个 metric 的话可以设置这个参数
        if not isinstance(metric_items, list):
            metric_items = [metric_items]
    
    # 这里是将推理生成的结果（列表）按照标准的 COCO 格式变成了 JSON 格式
    # 在 test.py 选择 --format-only 参数的话会直接调用这个函数返回结果
    # 只不过这里会将结果保存在一个临时文件夹中，因为我们并不需要保存这个结果
    # 如果是 bbox 测评的话，result_files 会得到两个字段保存的路径
    # {'bbox': '/tmp/tmpj126zjei/results.bbox.json', 
    # 'proposal': '/tmp/tmpj126zjei/results.bbox.json'}
    result_files, tmp_dir = self.format_results(results, jsonfile_prefix)
    
    # 这是 mmdet 额外加的记录 metric 的字典，在 eval 的最后一行会输出
    # OrderedDict([('bbox_mAP', x), ('bbox_mAP_50', x), ('bbox_mAP_75', x), ('bbox_mAP_s', x), ('bbox_mAP_m', x), ('bbox_mAP_l', x), ('bbox_mAP_copypaste', x)])
    eval_results = OrderedDict()
    # cocoGt 指的是从验证集中读取到的真实标签
    cocoGt = self.coco
    for metric in metrics:
        msg = f'Evaluating {metric}...'
        if logger is None:
            msg = '\n' + msg
        print_log(msg, logger=logger)
        
        # TODO
        if metric == 'proposal_fast':
            ar = self.fast_eval_recall(
                results, proposal_nums, iou_thrs, logger='silent')
            log_msg = []
            for i, num in enumerate(proposal_nums):
                eval_results[f'AR@{num}'] = ar[i]
                log_msg.append(f'\nAR@{num}\t{ar[i]:.4f}')
            log_msg = ''.join(log_msg)
            print_log(log_msg, logger=logger)
            continue

        iou_type = 'bbox' if metric == 'proposal' else metric
        if metric not in result_files:
            raise KeyError(f'{metric} is not in results')
        try:
            # 将刚刚保存的 COCO 标准结果给重新 load 进来
            predictions = mmcv.load(result_files[metric])
            if iou_type == 'segm':
                # Refer to https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocotools/coco.py#L331  # noqa
                # When evaluating mask AP, if the results contain bbox,
                # cocoapi will use the box area instead of the mask area
                # for calculating the instance area. Though the overall AP
                # is not affected, this leads to different
                # small/medium/large mask AP results.
                for x in predictions:
                    x.pop('bbox')
                warnings.simplefilter('once')
                warnings.warn(
                    'The key "bbox" is deleted for more accurate mask AP '
                    'of small/medium/large instances since v2.12.0. This '
                    'does not change the overall mAP calculation.',
                    UserWarning)
            # 通过 loadRes 将检测的结果加载为 cocoDt
            # 返回了一个 COCO 类的对象
            cocoDt = cocoGt.loadRes(predictions)
        except IndexError:
            print_log(
                'The testing results of the whole dataset is empty.',
                logger=logger,
                level=logging.ERROR)
            break
        
        # 直接调用 API 传入 cocoGt 和 cocoDt，以及评估的 iou 的方式
        cocoEval = COCOeval(cocoGt, cocoDt, iou_type)
        # 等同于 self.coco.get_cat_ids(cat_names=self.CLASSES)
        cocoEval.params.catIds = self.cat_ids
        # 等同于 self.coco.get_img_ids()
        cocoEval.params.imgIds = self.img_ids
        # 传进来的 proposal 参数，格式化成列表传入
        cocoEval.params.maxDets = list(proposal_nums)
        # 默认是从 0.5-0.95 的阈值，可以自己传入参数
        cocoEval.params.iouThrs = iou_thrs
        # mapping of cocoEval.stats
        coco_metric_names = {
            'mAP': 0,
            'mAP_50': 1,
            'mAP_75': 2,
            'mAP_s': 3,
            'mAP_m': 4,
            'mAP_l': 5,
            'AR@100': 6,
            'AR@300': 7,
            'AR@1000': 8,
            'AR_s@1000': 9,
            'AR_m@1000': 10,
            'AR_l@1000': 11
        }
        if metric_items is not None:
            for metric_item in metric_items:
                if metric_item not in coco_metric_names:
                    raise KeyError(
                        f'metric item {metric_item} is not supported')

        if metric == 'proposal':
            # 进行 AR 评估的话，不需要传入类别，默认用的是所有类别
            # CocoEval 的用法在后面会讲
            cocoEval.params.useCats = 0
            # 直接三步走，调用 API
            cocoEval.evaluate()
            cocoEval.accumulate()
            cocoEval.summarize()
            if metric_items is None:
                metric_items = [
                    'AR@100', 'AR@300', 'AR@1000', 'AR_s@1000',
                    'AR_m@1000', 'AR_l@1000'
                ]

            for item in metric_items:
                # cocoEval.stats 在 summarize() 之后就可以取到所有的 metric 的值
                # coco_metric_names 是为了跟 cocoEval.stats 的值做个映射方便取值
                # cocoEval.stats 有 12 个值，前 6 个是 AP 相关，后 6 个是 AR 相关
                val = float(
                    f'{cocoEval.stats[coco_metric_names[item]]:.3f}')
                eval_results[item] = val
        else:
            cocoEval.evaluate()
            cocoEval.accumulate()
            cocoEval.summarize()
            # 计算单个类别的 AP
            if classwise:  # Compute per-category AP
                # Compute per-category AP
                # from https://github.com/facebookresearch/detectron2/
                precisions = cocoEval.eval['precision']
                # precision: (iou, recall, cls, area range, max dets)
                assert len(self.cat_ids) == precisions.shape[2]

                results_per_category = []
                for idx, catId in enumerate(self.cat_ids):
                    # area range index 0: all area ranges
                    # max dets index -1: typically 100 per image
                    nm = self.coco.loadCats(catId)[0]
                    precision = precisions[:, :, idx, 0, -1]
                    precision = precision[precision > -1]
                    if precision.size:
                        ap = np.mean(precision)
                    else:
                        ap = float('nan')
                    results_per_category.append(
                        (f'{nm["name"]}', f'{float(ap):0.3f}'))

                num_columns = min(6, len(results_per_category) * 2)
                results_flatten = list(
                    itertools.chain(*results_per_category))
                headers = ['category', 'AP'] * (num_columns // 2)
                results_2d = itertools.zip_longest(*[
                    results_flatten[i::num_columns]
                    for i in range(num_columns)
                ])
                table_data = [headers]
                table_data += [result for result in results_2d]
                table = AsciiTable(table_data)
                print_log('\n' + table.table, logger=logger)

            if metric_items is None:
                metric_items = [
                    'mAP', 'mAP_50', 'mAP_75', 'mAP_s', 'mAP_m', 'mAP_l'
                ]

            for metric_item in metric_items:
                key = f'{metric}_{metric_item}'
                val = float(
                    f'{cocoEval.stats[coco_metric_names[metric_item]]:.3f}'
                )
                eval_results[key] = val
            # 前面 6 个是记录 AP 的，后面六个是 AR
            ap = cocoEval.stats[:6]
            # 改一下这里乘个 100 就可以生成百分号计数的 AP 了，不然总是小数
            eval_results[f'{metric}_mAP_copypaste'] = (
                f'{ap[0]:.3f} {ap[1]:.3f} {ap[2]:.3f} {ap[3]:.3f} '
                f'{ap[4]:.3f} {ap[5]:.3f}')
    # 细节，还将临时文件给清理了
    if tmp_dir is not None:
        tmp_dir.cleanup()
    return eval_results
```

## 小结



所以总结一下上述代码的总体逻辑：要评估 COCO 格式的数据集的话，首先需要产生推理结果，然后将推理结果进行格式化变成 COCO 格式的 json 列表，再通过读入验证集的真实标签，用 pycocotools 的 `loadRes`函数将推理结果进一步格式化成标准 COCO 对象。然后确定参数（评估 iou 的类型，得分阈值，评估方式等等），再直接调用 `cocoEval` 的 API 就得到了所有的 metric，一般保存在 `cocoEval` 的 `eval `和 `stats `变量中。因此可以用以下代码简单概括：

COCOeval 的 `loadRes` 函数可以接受字符串也可以接受已经读取好的 json 列表，我们可以直接传入 json 列表保存的路径，这个列表可以用 mmdet test 脚本的 `--format-only` 选项生成，然后就可以输出各种指标了 

```Python
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-g", "--gt", type=str, help="Assign the groud true path.", default=None)
    parser.add_argument("-d", "--dt", type=str, help="Assign the detection result path.", default=None)
    args = parser.parse_args()

    cocoGt = COCO(args.gt)
    cocoDt = cocoGt.loadRes(args.dt)
    cocoEval = COCOeval(cocoGt, cocoDt, "bbox")
    cocoEval.evaluate()
    cocoEval.accumulate()
    cocoEval.summarize()
```

# pycocotools/cocoeval.py

上面只从表面上介绍了一下评估的过程，可以看到 mmdet 在评估的代码中还是做了一些封装的，不过总之还是调用 pycocotools 的函数。这边我们就深入细节看看计算 mAP 的过程是怎么实现的。

## class COCOEval

### __init__



初始化函数写了一些 COCOEval 的使用方法，很贴心了

```Plain
The usage for CocoEval is as follows:
 cocoGt=..., cocoDt=...       # load dataset and results
 E = CocoEval(cocoGt,cocoDt); # initialize CocoEval object
 E.params.recThrs = ...;      # set parameters as desired
 E.evaluate();                # run per image evaluation
 E.accumulate();              # accumulate per image results
 E.summarize();               # display summary metrics of results
For example usage see evalDemo.m and http://mscoco.org/.

The evaluation parameters are as follows (defaults in brackets):
 imgIds     - [all] N img ids to use for evaluation
 catIds     - [all] K cat ids to use for evaluation
 iouThrs    - [.5:.05:.95] T=10 IoU thresholds for evaluation
 recThrs    - [0:.01:1] R=101 recall thresholds for evaluation
 areaRng    - [...] A=4 object area ranges for evaluation
 maxDets    - [1 10 100] M=3 thresholds on max detections per image
 iouType    - ['segm'] set iouType to 'segm', 'bbox' or 'keypoints'
 iouType replaced the now DEPRECATED useSegm parameter.
 useCats    - [1] if true use category labels for evaluation
Note: if useCats=0 category labels are ignored as in proposal scoring.
Note: multiple areaRngs [Ax2] and maxDets [Mx1] can be specified.

evaluate(): evaluates detections on every image and every category and
concats the results into the "evalImgs" with fields:
 dtIds      - [1xD] id for each of the D detections (dt)
 gtIds      - [1xG] id for each of the G ground truths (gt)
 dtMatches  - [TxD] matching gt id at each IoU or 0
 gtMatches  - [TxG] matching dt id at each IoU or 0
 dtScores   - [1xD] confidence of each dt
 gtIgnore   - [1xG] ignore flag for each gt
 dtIgnore   - [TxD] ignore flag for each dt at each IoU

accumulate(): accumulates the per-image, per-category evaluation
results in "evalImgs" into the dictionary "eval" with fields:
 params     - parameters used for evaluation
 date       - date evaluation was performed
 counts     - [T,R,K,A,M] parameter dimensions (see above)
 precision  - [TxRxKxAxM] precision for every evaluation setting
 recall     - [TxKxAxM] max recall for every evaluation setting
Note: precision and recall==-1 for settings with no gt objects.
```

### evaluate



这是评估三部曲的第一步，直接调用这个函数进行评估

```Python
def evaluate(self):
    '''
    Run per image evaluation on given images and store results (a list of dict) in self.evalImgs
    :return: None
    '''
    tic = time.time()
    print('Running per image evaluation...')
    # 可以用 p.__dict__ 以 dict 的形式获取到类的成员
    p = self.params
    # add backward compatibility if useSegm is specified in params
    if not p.useSegm is None:
        p.iouType = 'segm' if p.useSegm == 1 else 'bbox'
        print('useSegm (deprecated) is not None. Running {} evaluation'.format(p.iouType))
    print('Evaluate annotation type *{}*'.format(p.iouType))
    # numpy 转 list
    p.imgIds = list(np.unique(p.imgIds))
    if p.useCats:
        # numpy 转 list
        p.catIds = list(np.unique(p.catIds))
    p.maxDets = sorted(p.maxDets)
    self.params=p
    
    # 做一些准备工作，具体见下文
    self._prepare()
    # loop through images, area range, max detection number
    catIds = p.catIds if p.useCats else [-1]

    if p.iouType == 'segm' or p.iouType == 'bbox':
        computeIoU = self.computeIoU
    elif p.iouType == 'keypoints':
        computeIoU = self.computeOks
    # 对所有图片所有类别的图像都进行 IoU 的计算
    # computeIoU 得到的是对于同一张图片，gt 和 dt 的 IOU，所以得到的是 shape (#dt, #gt) 的 numpy 数组
    # 这个操作对每一个类别和每一张图片都要做一次，所以 self.ious 的 key 的长度为 #val_set*#classes, COCO 就是 5000*80
    self.ious = {(imgId, catId): computeIoU(imgId, catId) \
                    for imgId in p.imgIds
                    for catId in catIds}
    
    evaluateImg = self.evaluateImg
    # 取的是最大的那个，100，代表每一个类别最多保留的检测框的数量
    maxDet = p.maxDets[-1]
    # 进行单张图片单个类别的评估
    # 得到的是一个 #img*#class*#areaRng的列表，COCO 是 5000*80*4(ALl, Small, Medium, Large)
    # 每一个列表存储着该图片特定类别的 dt 和 gt 匹配结果
    self.evalImgs = [evaluateImg(imgId, catId, areaRng, maxDet)
             for catId in catIds
             for areaRng in p.areaRng
             for imgId in p.imgIds
         ]
     # 深拷贝一份参数，结束函数
    self._paramsEval = copy.deepcopy(self.params)
    toc = time.time()
    print('DONE (t={:0.2f}s).'.format(toc-tic))
```

### _prepare



这个函数做一些准备工作，定义一些变量

```Python
def _prepare(self):
    '''
    Prepare ._gts and ._dts for evaluation based on params
    :return: None
    '''
    def _toMask(anns, coco):
        # modify ann['segmentation'] by reference
        for ann in anns:
            rle = coco.annToRLE(ann)
            ann['segmentation'] = rle
    p = self.params
    # 如果 useCats 的话
    # 用户可以自己选择进行什么类别的评估
    if p.useCats:
        # 变成列表，每一个元素都是一个 json
        # 记录着对应的图片 id，坐标、得分、类别等等信息
        # ['image_id', 'bbox', 'score', 'category_id', 'segmentation', 'area', 'id', 'iscrowd']
        gts=self.cocoGt.loadAnns(self.cocoGt.getAnnIds(imgIds=p.imgIds, catIds=p.catIds))
        dts=self.cocoDt.loadAnns(self.cocoDt.getAnnIds(imgIds=p.imgIds, catIds=p.catIds))
    # 不用的话就不传入 catIds 参数，默认使用的是类别的结果
    else:
        gts=self.cocoGt.loadAnns(self.cocoGt.getAnnIds(imgIds=p.imgIds))
        dts=self.cocoDt.loadAnns(self.cocoDt.getAnnIds(imgIds=p.imgIds))

    # convert ground truth to mask if iouType == 'segm'
    if p.iouType == 'segm':
        _toMask(gts, self.cocoGt)
        _toMask(dts, self.cocoDt)
    # set ignore flag
    for gt in gts:
        gt['ignore'] = gt['ignore'] if 'ignore' in gt else 0
        gt['ignore'] = 'iscrowd' in gt and gt['iscrowd']
        if p.iouType == 'keypoints':
            gt['ignore'] = (gt['num_keypoints'] == 0) or gt['ignore']
    # defaultdict 里面传入一个工厂函数，如果获取不到 key 的话不会报错而是返回函数的默认值
    # 用在这里的话相当于图片中没有 gt 或 dt 的话就返回一个空的列表
    # 一开始列表就是空的，所以可以直接用 append，比较方便，这里的 key 就是（gt['image_id'], gt['category_id']）
    self._gts = defaultdict(list)       # gt for evaluation
    self._dts = defaultdict(list)       # dt for evaluation
    for gt in gts:
        self._gts[gt['image_id'], gt['category_id']].append(gt)
    for dt in dts:
        self._dts[dt['image_id'], dt['category_id']].append(dt)
    # 这里也同样，存放的 key 是每一张图的 id 以及每一个类别组成的 tuple    
    self.evalImgs = defaultdict(list)   # per-image per-category evaluation results
    # 这个是存放最终结果的变量
    self.eval     = {}                  # accumulated evaluation results
```

### computeIoU



单张图片单个类别的所有框两两计算 IoU，maxDets 参数限制了一个类别最多检测出来的框的数量，最终返回一个 numpy 数组，形状为 (#dt, #gt) 

```Python
def computeIoU(self, imgId, catId):
    p = self.params
    if p.useCats:
        gt = self._gts[imgId,catId]
        dt = self._dts[imgId,catId]
    else:
        gt = [_ for cId in p.catIds for _ in self._gts[imgId,cId]]
        dt = [_ for cId in p.catIds for _ in self._dts[imgId,cId]]
    if len(gt) == 0 and len(dt) ==0:
        return []
    inds = np.argsort([-d['score'] for d in dt], kind='mergesort')
    dt = [dt[i] for i in inds]
    if len(dt) > p.maxDets[-1]:
        dt=dt[0:p.maxDets[-1]]

    if p.iouType == 'segm':
        g = [g['segmentation'] for g in gt]
        d = [d['segmentation'] for d in dt]
    elif p.iouType == 'bbox':
        g = [g['bbox'] for g in gt]
        d = [d['bbox'] for d in dt]
    else:
        raise Exception('unknown iouType for iou computation')
```

### evaluateImg

这个函数进行单张图片单个类别的评估，里面根据几个条件使得检测框和 gt 匹配，得到匹配信息，是个比较重要的函数

```Python
def evaluateImg(self, imgId, catId, aRng, maxDet):
    '''
    perform evaluation for single category and image
    :return: dict (single image results)
    '''
    p = self.params
    if p.useCats:
        gt = self._gts[imgId,catId]
        dt = self._dts[imgId,catId]
    else:
        gt = [_ for cId in p.catIds for _ in self._gts[imgId,cId]]
        dt = [_ for cId in p.catIds for _ in self._dts[imgId,cId]]
    if len(gt) == 0 and len(dt) ==0:
        return None

    for g in gt:
        # 如果不是本次要匹配的 gt 对象的话，就给 _ignore 字段置 1
        # 因为根据面积来计算 AP 时得挑选出面积在范围内的 gt，超过范围的不会和 dt 进行匹配
        if g['ignore'] or (g['area']<aRng[0] or g['area']>aRng[1]):
            g['_ignore'] = 1
        else:
            g['_ignore'] = 0

    # sort dt highest score first, sort gt ignore last
    # 进行从小到大的排序，返回索引值，所以没被忽略的 gt 会排在前面
    gtind = np.argsort([g['_ignore'] for g in gt], kind='mergesort')
    gt = [gt[i] for i in gtind]
    # 从大到小排序，得分高的检测框在前面
    dtind = np.argsort([-d['score'] for d in dt], kind='mergesort')
    dt = [dt[i] for i in dtind[0:maxDet]]
    iscrowd = [int(o['iscrowd']) for o in gt]
    # load computed ious
    ious = self.ious[imgId, catId][:, gtind] if len(self.ious[imgId, catId]) > 0 else self.ious[imgId, catId]
    
    # 下面是重点，遍历 dt 和 gt 存储和每个 dt iou 最大的 gt 的索引
    # 同样也存储与每一个 gt iou 最大的 dt 索引
    T = len(p.iouThrs)
    G = len(gt)
    D = len(dt)
    gtm  = np.zeros((T,G))
    dtm  = np.zeros((T,D))
    gtIg = np.array([g['_ignore'] for g in gt])
    dtIg = np.zeros((T,D))
    if not len(ious)==0:
        for tind, t in enumerate(p.iouThrs):
            for dind, d in enumerate(dt):
                # information about best match so far (m=-1 -> unmatched)
                iou = min([t,1-1e-10])
                # 初始时候没有匹配，m=-1
                m   = -1
                for gind, g in enumerate(gt):
                    # 三种情况进行筛选
                    # if this gt already matched, and not a crowd, continue
                    # 这个 gt 已经被匹配上了，找下一个
                    if gtm[tind,gind]>0 and not iscrowd[gind]:
                        continue
                    # if dt matched to reg gt, and on ignore gt, stop
                    # 不明白这里为啥要 break 而不是 continue
                    # 明白了！！前面已经对 gtind 进行了排序，如果当前 gt 是 ignore 的话
                    # 后面的 gt 也一定是 ignore，所以直接跳过了，妙！
                    if m>-1 and gtIg[m]==0 and gtIg[gind]==1:
                        break
                    # continue to next gt unless better match made
                    # 找到和 dt 最大 iou 的那个 gt
                    if ious[dind,gind] < iou:
                        continue
                    # if match successful and best so far, store appropriately
                    # 存储临时变量
                    iou=ious[dind,gind]
                    m=gind
                # if match made， store id of match for both dt and gt
                if m ==-1:
                    continue
                # 如果匹配成功的话就保存各自匹配的 id，dt 保存的是匹配上的 gt 的 id，gt 保存的是匹配上的 dt 的 id
                # 所以没匹配上的 dtm 就是与目标框的IoU 小于阈值的，或者匹配的 gt 已经被其他框匹配了。
                # 没匹配上的框又不是忽略框的话会在后面被当成 false positive
                dtIg[tind,dind] = gtIg[m]
                dtm[tind,dind]  = gt[m]['id']
                gtm[tind,m]     = d['id']
    # set unmatched detections outside of area range to ignore
    # a 存储了超过面积范围的 dt
    a = np.array([d['area']<aRng[0] or d['area']>aRng[1] for d in dt]).reshape((1, len(dt)))
    # dtIg 存储了超过面积范围的 dt
    dtIg = np.logical_or(dtIg, np.logical_and(dtm==0, np.repeat(a,T,0)))
    # store results for given image and category
    # 所以这里就返回了一些信息，表示该图片中，对该类别的一些匹配情况，方便进行后续的分析
    return {
            'image_id':     imgId,
            'category_id':  catId,
            'aRng':         aRng,
            'maxDet':       maxDet,
            'dtIds':        [d['id'] for d in dt],
            'gtIds':        [g['id'] for g in gt],
            'dtMatches':    dtm,
            'gtMatches':    gtm,
            'dtScores':     [d['score'] for d in dt],
            'gtIgnore':     gtIg,
            'dtIgnore':     dtIg,
        }
```

### accumulate



这个函数对刚刚 `evaluate` 的中间结果进行累加，来求详细的评估指标。

```Python
def accumulate(self, p = None):
    '''
    Accumulate per image evaluation results and store the result in self.eval
    :param p: input params for evaluation
    :return: None
    '''
    print('Accumulating evaluation results...')
    tic = time.time()
    if not self.evalImgs:
        print('Please run evaluate() first')
    # allows input customized parameters
    if p is None:
        p = self.params    
    p.catIds = p.catIds if p.useCats == 1 else [-1]
    # 初始化一些变量，把 metrics 默认定义为 -1，
    T           = len(p.iouThrs)
    R           = len(p.recThrs)
    K           = len(p.catIds) if p.useCats else 1
    A           = len(p.areaRng)
    M           = len(p.maxDets)
    # 求出来的东西是针对每一个 iou 阈值，每一个 recall 阈值，每一个面积范围，每一个类别，每一个最大检测数量的列表
    precision   = -np.ones((T,R,K,A,M)) # -1 for the precision of absent categories
    recall      = -np.ones((T,K,A,M))
    scores      = -np.ones((T,R,K,A,M))

    # create dictionary for future indexing
    _pe = self._paramsEval
    catIds = _pe.catIds if _pe.useCats else [-1]
    setK = set(catIds)
    setA = set(map(tuple, _pe.areaRng))
    setM = set(_pe.maxDets)
    setI = set(_pe.imgIds)
    # get inds to evaluate

    k_list = [n for n, k in enumerate(p.catIds)  if k in setK]
    m_list = [m for n, m in enumerate(p.maxDets) if m in setM]
    # 所有面积范围列表
    a_list = [n for n, a in enumerate(map(lambda x: tuple(x), p.areaRng)) if a in setA]
    # 所有图片 id 列表
    i_list = [n for n, i in enumerate(p.imgIds)  if i in setI]
    I0 = len(_pe.imgIds)
    A0 = len(_pe.areaRng)
    # retrieve E at each category, area range, and max number of detections
    # 进行循环，求出每种情况下的指标
    for k, k0 in enumerate(k_list):
        Nk = k0*A0*I0
        for a, a0 in enumerate(a_list):
            Na = a0*I0
            for m, maxDet in enumerate(m_list):
                # self.evalImgs 得到的是一个 #img*#class*#areaRng的列表，所以要根据上面计算出来的偏移量取值
                # 一般取出来的 E 的长度就是验证集的图片数
                E = [self.evalImgs[Nk + Na + i] for i in i_list]
                E = [e for e in E if not e is None]
                if len(E) == 0:
                    continue
                # 把整个数据集的检测框的得分都拿出来进行从大到小的排序，shape: (#all_dets_in_the_dataset，)
                dtScores = np.concatenate([e['dtScores'][0:maxDet] for e in E])

                # different sorting method generates slightly different results.
                # mergesort is used to be consistent as Matlab implementation.
                inds = np.argsort(-dtScores, kind='mergesort')
                dtScoresSorted = dtScores[inds]
                # 同上，也是将整个数据集的结果进行合并，shape: (#iou_thres, #all_dets_in_the_dataset)
                dtm  = np.concatenate([e['dtMatches'][:,0:maxDet] for e in E], axis=1)[:,inds]
                dtIg = np.concatenate([e['dtIgnore'][:,0:maxDet]  for e in E], axis=1)[:,inds]
                # 同上，shape: (#all_gts_in_the_dataset，)
                gtIg = np.concatenate([e['gtIgnore'] for e in E])
                # 当前的 gt 数量，如果全是忽略的话，就跳过这一次评估
                npig = np.count_nonzero(gtIg==0 )
                if npig == 0:
                    continue
                # 计算出当前的 tp 和 fp
                tps = np.logical_and(               dtm,  np.logical_not(dtIg) )
                fps = np.logical_and(np.logical_not(dtm), np.logical_not(dtIg) )
                # 按行累加，统计出到当前索引为止的 tp 和 fp 的总数量
                tp_sum = np.cumsum(tps, axis=1).astype(dtype=float)
                fp_sum = np.cumsum(fps, axis=1).astype(dtype=float)
                # 对于每一个 iou 阈值来计算评估指标
                for t, (tp, fp) in enumerate(zip(tp_sum, fp_sum)):
                    tp = np.array(tp)
                    fp = np.array(fp)
                    # 所有图片的检测框的数量
                    nd = len(tp)
                    # 根据公式计算出每一次的 recall 和 precision
                    rc = tp / npig
                    pr = tp / (fp+tp+np.spacing(1))
                    # 存储结果，shape 为进行插值的 recall 的 size，也就是 [0.1: 0.01: 1] 的 101 个值
                    q  = np.zeros((R,))
                    ss = np.zeros((R,))
                    
                    # 召回率
                    if nd:
                        recall[t,k,a,m] = rc[-1]
                    else:
                        recall[t,k,a,m] = 0

                    # numpy is slow without cython optimization for accessing elements
                    # use python array gets significant speed improvement
                    pr = pr.tolist(); q = q.tolist()
                    
                    # 对 recall 进行插值（mAP 的计算是计算 pr 曲线插值的矩形的面积）
                    # COCO 的话就是利用 101 个 recall 值进行插值 [0.1: 0.01: 1]
                    # 其实最终的 mAP 就相当于所有 recall 点上的 precision 的 mean
                    # 因为默认是计算个各个矩形的面积相加，这里矩形的宽是 0.01，刚好可以当成 1/100，100 可以看成是 recall 插值点的个数
                    for i in range(nd-1, 0, -1):
                        if pr[i] > pr[i-1]:
                            pr[i-1] = pr[i]
                    
                    # 将 p.recThrs 按大小插入到 rc 左边，返回索引值
                    inds = np.searchsorted(rc, p.recThrs, side='left')
                    try:
                        # q 存储经过插值后的 precision 值
                        # ss 存储得分
                        for ri, pi in enumerate(inds):
                            q[ri] = pr[pi]
                            ss[ri] = dtScoresSorted[pi]
                    # 取不到的值就默认为 0
                    except:
                        pass
                    # 进行保存，shape (T,R,K,A,M)
                    # 分别代表 iou 阈值数量，recall 阈值数量，类别数，面积范围数，最大检测框数量
                    precision[t,:,k,a,m] = np.array(q)
                    scores[t,:,k,a,m] = np.array(ss)
    self.eval = {
        'params': p,
        'counts': [T, R, K, A, M],
        'date': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'precision': precision,
        'recall':   recall,
        'scores': scores,
    }
    toc = time.time()
    print('DONE (t={:0.2f}s).'.format( toc-tic))
```

### summarize



函数将上一步评估好的指标进行总结，以格式化输出。

```Python
def summarize(self):
    '''
    Compute and display summary metrics for evaluation results.
    Note this functin can *only* be applied on the default parameter setting
    '''
    def _summarize( ap=1, iouThr=None, areaRng='all', maxDets=100 ):
        p = self.params
        iStr = ' {:<18} {} @[ IoU={:<9} | area={:>6s} | maxDets={:>3d} ] = {:0.3f}'
        titleStr = 'Average Precision' if ap == 1 else 'Average Recall'
        typeStr = '(AP)' if ap==1 else '(AR)'
        iouStr = '{:0.2f}:{:0.2f}'.format(p.iouThrs[0], p.iouThrs[-1]) \
            if iouThr is None else '{:0.2f}'.format(iouThr)

        aind = [i for i, aRng in enumerate(p.areaRngLbl) if aRng == areaRng]
        mind = [i for i, mDet in enumerate(p.maxDets) if mDet == maxDets]
        if ap == 1:
            # dimension of precision: [TxRxKxAxM]
            s = self.eval['precision']
            # IoU
            if iouThr is not None:
                t = np.where(iouThr == p.iouThrs)[0]
                s = s[t]
            s = s[:,:,:,aind,mind]
        else:
            # dimension of recall: [TxKxAxM]
            s = self.eval['recall']
            if iouThr is not None:
                t = np.where(iouThr == p.iouThrs)[0]
                s = s[t]
            s = s[:,:,aind,mind]
        if len(s[s>-1])==0:
            mean_s = -1
        else:
            # 最终的 mAP 就相当于所有 recall 点上的 precision 的 mean
            # 因为默认是计算个各个矩形的面积相加，这里矩形的宽是 0.01，刚好可以当成 1/100，100 可以看成是 recall 插值点的个数
            mean_s = np.mean(s[s>-1])
        print(iStr.format(titleStr, typeStr, iouStr, areaRng, maxDets, mean_s))
        return mean_s
    def _summarizeDets():
        # 前面几个算的是 AP        
        stats = np.zeros((12,))
        stats[0] = _summarize(1)
        stats[1] = _summarize(1, iouThr=.5, maxDets=self.params.maxDets[2])
        stats[2] = _summarize(1, iouThr=.75, maxDets=self.params.maxDets[2])
        stats[3] = _summarize(1, areaRng='small', maxDets=self.params.maxDets[2])
        stats[4] = _summarize(1, areaRng='medium', maxDets=self.params.maxDets[2])
        stats[5] = _summarize(1, areaRng='large', maxDets=self.params.maxDets[2])
        # 后面几个算的是 recall
        stats[6] = _summarize(0, maxDets=self.params.maxDets[0])
        stats[7] = _summarize(0, maxDets=self.params.maxDets[1])
        stats[8] = _summarize(0, maxDets=self.params.maxDets[2])
        stats[9] = _summarize(0, areaRng='small', maxDets=self.params.maxDets[2])
        stats[10] = _summarize(0, areaRng='medium', maxDets=self.params.maxDets[2])
        stats[11] = _summarize(0, areaRng='large', maxDets=self.params.maxDets[2])
        return stats
    def _summarizeKps():
        stats = np.zeros((10,))
        stats[0] = _summarize(1, maxDets=20)
        stats[1] = _summarize(1, maxDets=20, iouThr=.5)
        stats[2] = _summarize(1, maxDets=20, iouThr=.75)
        stats[3] = _summarize(1, maxDets=20, areaRng='medium')
        stats[4] = _summarize(1, maxDets=20, areaRng='large')
        stats[5] = _summarize(0, maxDets=20)
        stats[6] = _summarize(0, maxDets=20, iouThr=.5)
        stats[7] = _summarize(0, maxDets=20, iouThr=.75)
        stats[8] = _summarize(0, maxDets=20, areaRng='medium')
        stats[9] = _summarize(0, maxDets=20, areaRng='large')
        return stats
    if not self.eval:
        raise Exception('Please run accumulate() first')
    iouType = self.params.iouType
    if iouType == 'segm' or iouType == 'bbox':
        summarize = _summarizeDets
    elif iouType == 'keypoints':
        summarize = _summarizeKps
    self.stats = summarize()

def __str__(self):
    self.summarize()
```

## class Params



这是 COCOEval 默认的参数结构体，主要是根据传进来的 `iouType` 参数来调用相应的参数初始化函数，

```Python
class Params:
    '''
    Params for coco evaluation api
    '''
    def setDetParams(self):
        self.imgIds = []
        self.catIds = []
        # np.arange causes trouble.  the data point on arange is slightly larger than the true value
        self.iouThrs = np.linspace(.5, 0.95, int(np.round((0.95 - .5) / .05)) + 1, endpoint=True)
        self.recThrs = np.linspace(.0, 1.00, int(np.round((1.00 - .0) / .01)) + 1, endpoint=True)
        # 用来计算 AR 的参数
        self.maxDets = [1, 10, 100]
        # 面积的范围，分别对应所有、小目标、中目标、大目标
        self.areaRng = [[0 ** 2, 1e5 ** 2], [0 ** 2, 32 ** 2], [32 ** 2, 96 ** 2], [96 ** 2, 1e5 ** 2]]
        self.areaRngLbl = ['all', 'small', 'medium', 'large']
        # 是否用到类别标签，计算 Recall 的话会置 0
        self.useCats = 1

    def setKpParams(self):
        self.imgIds = []
        self.catIds = []
        # np.arange causes trouble.  the data point on arange is slightly larger than the true value
        self.iouThrs = np.linspace(.5, 0.95, int(np.round((0.95 - .5) / .05)) + 1, endpoint=True)
        self.recThrs = np.linspace(.0, 1.00, int(np.round((1.00 - .0) / .01)) + 1, endpoint=True)
        self.maxDets = [20]
        self.areaRng = [[0 ** 2, 1e5 ** 2], [32 ** 2, 96 ** 2], [96 ** 2, 1e5 ** 2]]
        self.areaRngLbl = ['all', 'medium', 'large']
        self.useCats = 1
        self.kpt_oks_sigmas = np.array([.26, .25, .25, .35, .35, .79, .79, .72, .72, .62,.62, 1.07, 1.07, .87, .87, .89, .89])/10.0

    def __init__(self, iouType='segm'):
        if iouType == 'segm' or iouType == 'bbox':
            self.setDetParams()
        elif iouType == 'keypoints':
            self.setKpParams()
        else:
            raise Exception('iouType not supported')
        self.iouType = iouType
        # useSegm is deprecated
        self.useSegm = None
```

## 小结



以上就是 COCOEval 对检测结果的评估方式，也就是调用三板斧就能得到最终的结果。完全掌握了上述的过程之后我们就可以按照我们想要的方式对结果进行评估了，也就是在评估之前传入指定的参数（iou 阈值、类别、面积范围、最大检测数量等等），我们的结果主要保存在 stats 以及 precision 里面，前者是一个长度为 11 的数组，每一个元素的意义都是固定的，后者是一个 5 维的向量，我们可以传入参数来得到我们想要的具体维度的结果。

# 自定义评估方式

比如 mmdet 评估 COCO 有给到一个 `classwise`选项，把这个选项打开之后呢可以看到每一个类别的 AP，但是默认是 mAP，假如我们想要 AP50 的结果就可以修改此处的代码，下面是我修改之后的，把所有的 AP 都计算出来了。

```Python
if classwise:  # Compute per-category AP
    # Compute per-category AP
    # from https://github.com/facebookresearch/detectron2/
    precisions = cocoEval.eval['precision']
    # precision: (iou, recall, cls, area range, max dets)
    assert len(self.cat_ids) == precisions.shape[2]

    results_per_category = []
    for idx, catId in enumerate(self.cat_ids):
        t = []
        # area range index 0: all area ranges
        # max dets index -1: typically 100 per image
        nm = self.coco.loadCats(catId)[0]
        precision = precisions[:, :, idx, 0, -1]
        precision = precision[precision > -1]
        if precision.size:
            ap = np.mean(precision)
        else:
            ap = float('nan')
        t.append(f'{nm["name"]}')
        t.append(f'{float(ap):0.3f}')
        
        for iou in [0, 5]:
            precision = precisions[iou, :, idx, 0, -1]
            precision = precision[precision > -1]
            if precision.size:
                ap = np.mean(precision)
            else:
                ap = float('nan')
            t.append(f'{float(ap):0.3f}')
        
        for area in [1, 2, 3]:
            precision = precisions[:, :, idx, area, -1]
            precision = precision[precision > -1]
            if precision.size:
                ap = np.mean(precision)
            else:
                ap = float('nan')
            t.append(f'{float(ap):0.3f}')
        results_per_category.append(tuple(t))

    num_columns = len(results_per_category[0])
    results_flatten = list(
        itertools.chain(*results_per_category))
    headers = ['category', 'mAP', 'mAP_50', 'mAP_75', 'mAP_s', 'mAP_m', 'mAP_l']
    results_2d = itertools.zip_longest(*[
        results_flatten[i::num_columns]
        for i in range(num_columns)
    ])
    table_data = [headers]
    table_data += [result for result in results_2d]
    table = AsciiTable(table_data)
    print_log('\n' + table.table, logger=logger)
```

对比一下，这是之前默认的表格，只显示出每一个类别的 mAP 的值。

![img](https://user-images.githubusercontent.com/33142987/228175045-233bdf3e-4449-40b3-988e-456895edff6f.png)

这是我修改之后的，每个类别的所有 AP 值都显示出来了。

![img](https://user-images.githubusercontent.com/33142987/228175111-b0676f02-30bc-4358-9ef7-fdcbcbd0adcf.png)