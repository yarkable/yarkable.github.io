---
layout:     post
title:      ByteTrack注释详解
subtitle:   
date:       2022-07-17
author:     kevin
header-img: img/green-bg.jpg
catalog: true
tags:
    - deep learning
    - MOT

---



## preface



最近有用到多目标追踪 Multi Object Tracking 的东西，看过了经典的 DeepSort 源码之后觉得 tracking 挺有意思的也挺有挑战的，ByteTrack 是多目标追踪里面一个相对比较新的的追踪器 （ECCV2022），也比较简单，这里就对源码做一些注释，仅供日后复习参考。



ByteTrack 是 TBD(Tracking By Detection) 的方法，每一帧都需要进行检测，然后通过卡尔曼预测出每一条轨迹在当前帧的位置，通过所有轨迹和检测框进行关联给每一条轨迹找到对应的检测框，结合真实的检测框对每段轨迹的卡尔曼预测器的均值和方差进行调整。



其中，卡尔曼预测器的作用就是预测出轨迹在当前帧可能出现的位置，一般在代码中会有两个函数，一个是 `predict`，一个是 `update`，其中 predict 用来预测位置，update 的作用是根据预测的位置和配对的检测框对卡尔曼的参数做调整。同样，Tracker 也有个 update 函数，这个函数就相当于检测领域的 detect 函数，调用之后会返回每一帧轨迹（追踪到的，丢弃的，消失的）



[ByteTrack](https://github.com/ifzhang/ByteTrack) 主要的文件只有 4 个，其中主要的是 `byte_tracker.py`，里面包含了 Tracker 的逻辑以及每一段 tracklet 的成员信息。在这里说点预备知识，一段轨迹也就是 tracklet 是由很多个 box 组成的时序上的序列，其实就是某一个 id 在画面中按时序出现的位置；并且 ByteTrack 其实是不用训练的，只要在数据集上训练好检测模型就行了，TBD 形式的追踪器实际上就是对检测结果进行一些逻辑处理。



## basetrack.py



```python
import numpy as np
from collections import OrderedDict

# tracker 的 4 种状态
class TrackState(object):
    New = 0
    Tracked = 1
    Lost = 2
    Removed = 3

# tracklet 的基类，拥有一段轨迹的各种属性，包括 id，当前出现的 frame_id 等等
class BaseTrack(object):
    _count = 0

    track_id = 0
    is_activated = False
    state = TrackState.New

    history = OrderedDict()
    features = []
    curr_feature = None
    score = 0
    start_frame = 0
    frame_id = 0
    time_since_update = 0
...

```

## kalman_filter.py



这个函数就是关于卡尔曼滤波器的一些函数，我们将物体的运动假设为匀速运动，运用卡尔曼滤波器对物体在下一帧图像中出现的位置进行一个预测。他包含 8 个状态量（x, y, a, h, vx, vy, va, vh），分别是 bbox 的中心点坐标、 bbox 宽高比例、bbox 的高，以及对应的速度，这里只简单罗列一下，想了解更多的话建议去看这个[知乎回答](https://zhuanlan.zhihu.com/p/90835266)

```python

class KalmanFilter(object):
    """
    A simple Kalman filter for tracking bounding boxes in image space.

    The 8-dimensional state space

        x, y, a, h, vx, vy, va, vh

    contains the bounding box center position (x, y), aspect ratio a, height h,
    and their respective velocities.

    Object motion follows a constant velocity model. The bounding box location
    (x, y, a, h) is taken as direct observation of the state space (linear
    observation model).

    """

    def __init__(self):
        ndim, dt = 4, 1.

        # Create Kalman filter model matrices.
        self._motion_mat = np.eye(2 * ndim, 2 * ndim)
        for i in range(ndim):
            self._motion_mat[i, ndim + i] = dt
        self._update_mat = np.eye(ndim, 2 * ndim)

        # Motion and observation uncertainty are chosen relative to the current
        # state estimate. These weights control the amount of uncertainty in
        # the model. This is a bit hacky.
        self._std_weight_position = 1. / 20
        self._std_weight_velocity = 1. / 160

    def initiate(self, measurement):
        """Create track from unassociated measurement.

        Parameters
        ----------
        measurement : ndarray
            Bounding box coordinates (x, y, a, h) with center position (x, y),
            aspect ratio a, and height h.

        Returns
        -------
        (ndarray, ndarray)
            Returns the mean vector (8 dimensional) and covariance matrix (8x8
            dimensional) of the new track. Unobserved velocities are initialized
            to 0 mean.

        """        
        return mean, covariance

    def predict(self, mean, covariance):
        """Run Kalman filter prediction step.

        Parameters
        ----------
        mean : ndarray
            The 8 dimensional mean vector of the object state at the previous
            time step.
        covariance : ndarray
            The 8x8 dimensional covariance matrix of the object state at the
            previous time step.

        Returns
        -------
        (ndarray, ndarray)
            Returns the mean vector and covariance matrix of the predicted
            state. Unobserved velocities are initialized to 0 mean.

        """        
        return mean, covariance

    def project(self, mean, covariance):       
        return mean, covariance + innovation_cov

    def multi_predict(self, mean, covariance):
        """Run Kalman filter prediction step (Vectorized version).
        Parameters
        ----------
        mean : ndarray
            The Nx8 dimensional mean matrix of the object states at the previous
            time step.
        covariance : ndarray
            The Nx8x8 dimensional covariance matrics of the object states at the
            previous time step.
        Returns
        -------
        (ndarray, ndarray)
            Returns the mean vector and covariance matrix of the predicted
            state. Unobserved velocities are initialized to 0 mean.
        """
        return mean, covariance

    def update(self, mean, covariance, measurement):
        """Run Kalman filter correction step.

        Parameters
        ----------
        mean : ndarray
            The predicted state's mean vector (8 dimensional).
        covariance : ndarray
            The state's covariance matrix (8x8 dimensional).
        measurement : ndarray
            The 4 dimensional measurement vector (x, y, a, h), where (x, y)
            is the center position, a the aspect ratio, and h the height of the
            bounding box.

        Returns
        -------
        (ndarray, ndarray)
            Returns the measurement-corrected state distribution.

        """     
        return new_mean, new_covariance

    def gating_distance(self, mean, covariance, measurements,
                        only_position=False, metric='maha'):       
```



## byte_tracker.py



