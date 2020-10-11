---
layout: post
title: pycocotools一些api的用法记录
subtitle: 
date: 2020-10-10
author: kevin
header-img: img/green-bg.jpg
catalog: true
tags:
    - deep learning
    - object detection
---



## preface



好多目标检测论文都用的是 COCO 的数据集，所以代码也是基于 COCO，因为这个数据集本身比较复杂，为了简化研究人员在处理数据集代码上面花费的时间，有个叫做 pycocotools 的 python 包出现了，在很多论文开源代码中大家都是用这个包来进行代码复现的，因此这里就来小小总结一下



首先需要引入这个包，然后创建一个 coco 对象，创建时要将 annotation 的路径传进去

```python
from pycocotools.coco import COCO
coco = COCO('/NAS_REMOTE/PUBLIC/data/coco2017/annotations/instances_train2017.json')
```

如果直接运行的话就会输出以下  log 

```txt
loading annotations into memory...
Done (t=15.79s)
creating index...
index created!
```



创建好了 coco 对象后就可以使用他的一些 API 了，下面列举一些常用的，后续有用到其他的话也会继续更新



## getImgIds



`coco.getImgIds(imgIds=[], catIds=[])` 可以获取到 coco 所有图片对应的 id 号，以便后续处理，另外，传入 catIds 参数的话也可以只返回特定的类别的图片对应的 id 号，但是 catIds 需要通过其他方式获得，假设我们知道狗的 catId 为 18，我们就可以通过下面代码得到所有有狗的图片的 id 号

```python
coco.getImgIds(catIds=18)
```

> [98304, 204800, 524291, 311301, 491525, 147471, 131087, 278550, 581654, 253981, 450590, 106525, 368676, 253988, ...]



## getCatIds



`getCatIds(catNms=[], supNms=[], catIds=[])` 可以获取到 coco 类别对应的 label 号，因为 coco 有 80 类，所以如果不传入其他参数的话返回的就是一个拥有 80 个元素的列表（不是连续的数字，中间会跳过几个数字

```python
coco.getCatIds()
```

> [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, ...]

当然也可以传入类别名字，筛选出特定类别代表的 label ，如下说明 dog 的标签为 18

```python
coco.getCatIds(catNms='dog')
```

> [18]



## getAnnIds



`getAnnIds(imgIds=[], catIds=[], areaRng=[], iscrowd=None)` 获取某图像对应的 annotation 的 id 号，即 groundtruth，以下说明了该图有 4 个 标注

```python
coco.getAnnIds(imgIds=coco.getImgIds()[0])
```

> [151091, 202758, 1260346, 1766676]



## loadImgs



`loadImgs(ids=[])` 用的时候会在里面加上图片对应的 id 号返回某张图片的信息

```python
coco.loadImgs(ids=coco.getImgIds()[0])
```

> [{'coco_url': 'http://images.cocoda...391895.jpg', 'date_captured': '2013-11-14 11:18:45', 'file_name': '000000391895.jpg', 'flickr_url': 'http://farm9.staticf...8349_z.jpg', 'height': 360, 'id': 391895, 'license': 3, 'width': 640}]



## loadCats



`loadCats(ids=[])` 用法和上面差不多，是通过类别的 label 来返回className

```python
coco.loadCats(ids=18)
```

> [{'id': 18, 'name': 'dog', 'supercategory': 'animal'}]



## loadAnns



`loadAnns(ids=[])` 用来加载特定 annotation id 号的标注

```python
coco.loadAnns(ids=202758)
```

> [{'area': 14107.271300000002, 'bbox': [...], 'category_id': 1, 'id': 202758, 'image_id': 391895, 'iscrowd': 0, 'segmentation': [...]}]



## showAnns



`showAnns(anns, draw_bbox=False)`  同上，如果获取到了 Anno 信息之后，可以用这个函数直接将 Anno 信息给可视化出来，挺有意思的

```python
dataset_dir = '/NAS_REMOTE/PUBLIC/data/coco2017/train2017/'
coco = COCO(os.path.join('/NAS_REMOTE/PUBLIC/data/coco2017', 'annotations', 'instances_' + 'train2017' + '.json'))
image_ids = coco.getImgIds()
img = coco.loadImgs(image_ids[0])[0]
I = io.imread(dataset_dir + img['file_name'])
plt.axis('off')
plt.imshow(I)
annIds = coco.getAnnIds(imgIds=image_ids[0])
annos = coco.loadAnns(ids=annIds)
coco.showAnns(annos)
plt.show()
```



![mask](https://i.loli.net/2020/10/11/P1RFMyp8alKCJXH.png)



还可以在 `showAnns` 中将 `draw_bbox` 参数变成 `True` 来画出 bbox



![图片.png](https://i.loli.net/2020/10/11/pHq6IXQNvVB4G7z.png)

## 实际应用



PyTorch 中一般将数据集封装为一个类，继承 Dataset 父类，一次返回图像和其 label，其实就是在上述 API 基础上进行一些改动，主要是 loadImgs 和 loadAnns 这两个函数，下面截取 [pytorch-retinanet](https://github.com/yhenon/pytorch-retinanet) 中的部分代码作为示范

```python
def __getitem__(self, idx):

    img = self.load_image(idx)
    annot = self.load_annotations(idx)
    sample = {'img': img, 'annot': annot}
    if self.transform:
        sample = self.transform(sample)

    return sample

def load_image(self, image_index):
    image_info = self.coco.loadImgs(self.image_ids[image_index])[0]
    path       = os.path.join(self.root_dir, self.set_name, image_info['file_name'])
    img = skimage.io.imread(path)

    if len(img.shape) == 2:
        img = skimage.color.gray2rgb(img)

    return img.astype(np.float32)/255.0

def load_annotations(self, image_index):
    # get ground truth annotations
    annotations_ids = self.coco.getAnnIds(imgIds=self.image_ids[image_index], iscrowd=False)
    annotations     = np.zeros((0, 5))

    # some images appear to miss annotations (like image with id 257034)
    if len(annotations_ids) == 0:
        return annotations

    # parse annotations
    coco_annotations = self.coco.loadAnns(annotations_ids)
    for idx, a in enumerate(coco_annotations):

        # some annotations have basically no width / height, skip them
        if a['bbox'][2] < 1 or a['bbox'][3] < 1:
            continue
        
        # 4 个 bbox 坐标，一个 id 号
        annotation        = np.zeros((1, 5))
        annotation[0, :4] = a['bbox']
        annotation[0, 4]  = self.coco_label_to_label(a['category_id'])
        annotations       = np.append(annotations, annotation, axis=0)

    # transform from [x, y, w, h] to [x1, y1, x2, y2]
    annotations[:, 2] = annotations[:, 0] + annotations[:, 2]
    annotations[:, 3] = annotations[:, 1] + annotations[:, 3]

    return annotations
```



## reference



https://github.com/yhenon/pytorch-retinanet

https://blog.csdn.net/u013832707/article/details/94445495

