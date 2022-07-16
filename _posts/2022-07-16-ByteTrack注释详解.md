---
layout:     post
title:      ByteTrack注释详解
subtitle:   
date:       2022-07-16
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



每一帧的主要逻辑（非常经典，建议背诵）：

> 检测器得到 bbox → 卡尔曼滤波对 track 进行预测 → 使用匈牙利算法将预测后的 tracks 和当前帧中的 detecions 进行匹配（IOU匹配） → 卡尔曼滤波状态更新

```python
import numpy as np
from collections import deque
import os
import os.path as osp
import copy
import torch
import torch.nn.functional as F

from .kalman_filter import KalmanFilter
from yolox.tracker import matching
from .basetrack import BaseTrack, TrackState

# 继承 BaseTrack 的单个 track 类
class STrack(BaseTrack):
    shared_kalman = KalmanFilter()
    def __init__(self, tlwh, score):

        # wait activate
        # 初始化 track 全部都是 False 的状态
        # 一般是第一次出现某个 track 的情景
        self._tlwh = np.asarray(tlwh, dtype=np.float)
        self.kalman_filter = None
        self.mean, self.covariance = None, None
        self.is_activated = False

        self.score = score
        self.tracklet_len = 0
	
  	# 预测这个 track 下一次的位置，其实就是调用自身卡尔曼的 predict 函数更新均值和方差
    def predict(self):
        mean_state = self.mean.copy()
        if self.state != TrackState.Tracked:
            mean_state[7] = 0
        self.mean, self.covariance = self.kalman_filter.predict(mean_state, self.covariance)

    @staticmethod
    # 这个就是 predict 函数的矩阵版本，做的事情是一样的
    def multi_predict(stracks):
        if len(stracks) > 0:
            multi_mean = np.asarray([st.mean.copy() for st in stracks])
            multi_covariance = np.asarray([st.covariance for st in stracks])
            for i, st in enumerate(stracks):
                if st.state != TrackState.Tracked:
                    multi_mean[i][7] = 0
            multi_mean, multi_covariance = STrack.shared_kalman.multi_predict(multi_mean, multi_covariance)
            for i, (mean, cov) in enumerate(zip(multi_mean, multi_covariance)):
                stracks[i].mean = mean
                stracks[i].covariance = cov
		
    # 新激活一个轨迹，用轨迹的初始框来初始化对应的卡尔曼滤波器的参数，并且记录下 track 的 id
    # 这个是新建一个 track 调用的函数，并且如果是视频刚开始的话，直接会将 track 的状态变成激活态
    # 不是在视频刚开始激活的框的状态为未激活，需要下一帧还有检测框与其进行匹配才会变成激活状态
    def activate(self, kalman_filter, frame_id):
        """Start a new tracklet"""
        self.kalman_filter = kalman_filter
        self.track_id = self.next_id()
        self.mean, self.covariance = self.kalman_filter.initiate(self.tlwh_to_xyah(self._tlwh))

        self.tracklet_len = 0
        self.state = TrackState.Tracked
        if frame_id == 1:
            self.is_activated = True
        # self.is_activated = True
        self.frame_id = frame_id
        self.start_frame = frame_id
		
    # 这个应该是轨迹被遮挡或者消失之后重新激活轨迹调用的函数
    def re_activate(self, new_track, frame_id, new_id=False):
        self.mean, self.covariance = self.kalman_filter.update(
            self.mean, self.covariance, self.tlwh_to_xyah(new_track.tlwh)
        )
        self.tracklet_len = 0
        self.state = TrackState.Tracked
        self.is_activated = True
        self.frame_id = frame_id
        if new_id:
            self.track_id = self.next_id()
        self.score = new_track.score
		
    # 更新轨迹的位置
    def update(self, new_track, frame_id):
        """
        Update a matched track
        :type new_track: STrack
        :type frame_id: int
        :type update_feature: bool
        :return:
        """
        self.frame_id = frame_id
        self.tracklet_len += 1

        new_tlwh = new_track.tlwh
        self.mean, self.covariance = self.kalman_filter.update(
            self.mean, self.covariance, self.tlwh_to_xyah(new_tlwh))
        self.state = TrackState.Tracked
        self.is_activated = True

        self.score = new_track.score

    @property
    # @jit(nopython=True)
    # 这个函数很重要，在进行匹配的时候会调用到他，指的是 track 在经过卡尔曼预测之后在当前帧的位置
    # 所以这里用了 mean，因为卡尔曼经过 predict 之后会更新 mean 和 covariance 的状态，mean 是
    # [cx, cy, a, h, vx, vy, va, vh]，所以 self.mean[:4] 指的就是预测框的坐标信息
    def tlwh(self):
        """Get current position in bounding box format `(top left x, top left y,
                width, height)`.
        """
        if self.mean is None:
            return self._tlwh.copy()
        ret = self.mean[:4].copy()
        ret[2] *= ret[3]
        ret[:2] -= ret[2:] / 2
        return ret

    @property
    # @jit(nopython=True)
    def tlbr(self):
        """Convert bounding box to format `(min x, min y, max x, max y)`, i.e.,
        `(top left, bottom right)`.
        """
        ret = self.tlwh.copy()
        ret[2:] += ret[:2]
        return ret

class BYTETracker(object):
    def __init__(self, args, frame_rate=30):
        self.tracked_stracks = []  # type: list[STrack]
        self.lost_stracks = []  # type: list[STrack]
        self.removed_stracks = []  # type: list[STrack]

        self.frame_id = 0
        self.args = args
        #self.det_thresh = args.track_thresh
        self.det_thresh = args.track_thresh + 0.1
        # 缓冲的帧数，超过这么多帧丢失目标才算是真正丢失
        self.buffer_size = int(frame_rate / 30.0 * args.track_buffer)
        self.max_time_lost = self.buffer_size
        self.kalman_filter = KalmanFilter()
	
  	# 追踪主要逻辑函数
    def update(self, output_results, img_info, img_size):
        self.frame_id += 1
        activated_starcks = []
        refind_stracks = []
        lost_stracks = []
        removed_stracks = []
				
        # output_results 是 [xyxy,score] 或者 [xyxy, score, conf] 的情况
        if output_results.shape[1] == 5:
            scores = output_results[:, 4]
            bboxes = output_results[:, :4]
        else:
            output_results = output_results.cpu().numpy()
            scores = output_results[:, 4] * output_results[:, 5]
            bboxes = output_results[:, :4]  # x1y1x2y2
        img_h, img_w = img_info[0], img_info[1]
        scale = min(img_size[0] / float(img_h), img_size[1] / float(img_w))
        bboxes /= scale
				
        # 找到置信度高的框，作为第一次关联的框
        remain_inds = scores > self.args.track_thresh
        inds_low = scores > 0.1
        inds_high = scores < self.args.track_thresh
				# 找到置信度低的框，作为第二次关联的框
        inds_second = np.logical_and(inds_low, inds_high)
        dets_second = bboxes[inds_second]
        dets = bboxes[remain_inds]
        scores_keep = scores[remain_inds]
        scores_second = scores[inds_second]

        if len(dets) > 0:
            '''Detections'''
            # 把初始框封装成 STrack 的格式
            detections = [STrack(STrack.tlbr_to_tlwh(tlbr), s) for
                          (tlbr, s) in zip(dets, scores_keep)]
        else:
            detections = []

        ''' Add newly detected tracklets to tracked_stracks'''
        unconfirmed = []
        tracked_stracks = []  # type: list[STrack]
        for track in self.tracked_stracks:
          	# 当 track 只有一帧的记录时，is_activated=False
            if not track.is_activated:
                unconfirmed.append(track)
            else:
                tracked_stracks.append(track)

        ''' Step 2: First association, with high score detection boxes'''
        # 将已经追踪到的 track 和丢失的 track 合并
        # 丢失的 track 代表某一帧可能丢了一次，但是仍然在缓冲帧范围之内，所以依然可以用来匹配
        strack_pool = joint_stracks(tracked_stracks, self.lost_stracks)
        # Predict the current location with KF
        # 先用卡尔曼预测每一条轨迹在当前帧的位置
        STrack.multi_predict(strack_pool)
        # 让预测后的 track 和当前帧的 detection 框做 cost_matrix，用的方式为 IOU 关联
        # 这里的 iou_distance 函数中调用了 track.tlbr，返回的是预测之后的 track 坐标信息
        dists = matching.iou_distance(strack_pool, detections)
        if not self.args.mot20:
            dists = matching.fuse_score(dists, detections)
        # 用匈牙利算法算出相匹配的 track 和 detection 的索引，以及没有被匹配到的 track 和没有被匹配到的 detection 框的索引
        matches, u_track, u_detection = matching.linear_assignment(dists, thresh=self.args.match_thresh)

        for itracked, idet in matches:
          	# 找到匹配到的所有 track&detection pair 并且用 detection 来更新卡尔曼的状态
            track = strack_pool[itracked]
            det = detections[idet]
            # 对应 strack_pool 中的 tracked_stracks
            if track.state == TrackState.Tracked:
                track.update(detections[idet], self.frame_id)
                activated_starcks.append(track)
            # 对应 strack_pool 中的 self.lost_stracks，重新激活 track
            else:
                track.re_activate(det, self.frame_id, new_id=False)
                refind_stracks.append(track)

        ''' Step 3: Second association, with low score detection boxes'''
        # association the untrack to the low score detections
        if len(dets_second) > 0:
            '''Detections'''
             = [STrack(STrack.tlbr_to_tlwh(tlbr), s) for
                          (tlbr, s) in zip(dets_second, scores_second)]
        else:
            detections_second = []
        # 找出 strack_pool 中没有被匹配到的 track（这帧目标被遮挡的情况）
        r_tracked_stracks = [strack_pool[i] for i in u_track if strack_pool[i].state == TrackState.Tracked]
        dists = matching.iou_distance(r_tracked_stracks, detections_second)
        matches, u_track, u_detection_second = matching.linear_assignment(dists, thresh=0.5)
        # 在低置信度的检测框中再次与没有被匹配到的 track 做 IOU 匹配
        for itracked, idet in matches:
            track = r_tracked_stracks[itracked]
            det = detections_second[idet]
            if track.state == TrackState.Tracked:
                track.update(det, self.frame_id)
                activated_starcks.append(track)
            else:
                track.re_activate(det, self.frame_id, new_id=False)
                refind_stracks.append(track)
				
        # 如果 track 经过两次匹配之后还没有匹配到 box 的话，就标记为丢失了
        for it in u_track:
            track = r_tracked_stracks[it]
            if not track.state == TrackState.Lost:
                track.mark_lost()
                lost_stracks.append(track)

        '''Deal with unconfirmed tracks, usually tracks with only one beginning frame'''
        # 处理第一次匹配时没有被 track 匹配的检测框（一般是这个检测框第一次出现的情形）
        detections = [detections[i] for i in u_detection]
        # 计算未被匹配的框和不确定的 track 之间的 cost_matrix
        dists = matching.iou_distance(unconfirmed, detections)
        if not self.args.mot20:
            dists = matching.fuse_score(dists, detections)
        matches, u_unconfirmed, u_detection = matching.linear_assignment(dists, thresh=0.7)
        # 如果能够匹配上的话，说明这个 track 已经是确定状态了，
        # 用当前匹配到的框对卡尔曼的预测进行调节，并且将其加入到 activated_starcks
        for itracked, idet in matches:
            unconfirmed[itracked].update(detections[idet], self.frame_id)
            activated_starcks.append(unconfirmed[itracked])
        # 匹配不上的 unconfirmed_track 就直接删除了，说明这个 track 只出现了一帧
        for it in u_unconfirmed:
            track = unconfirmed[it]
            track.mark_removed()
            removed_stracks.append(track)

        """ Step 4: Init new stracks"""
        # 经过上面这些步骤之后，如果还有没被匹配的检测框，说明可能画面中新来了一个物体
        # 那么就直接将他视为一个新的 track，但是这个 track 的状态并不是激活态
        # 在下一次循环的时候会先将他放到 unconfirmed_track 中去，然后根据有没有框匹配他来决定是激活还是丢弃
        for inew in u_detection:
            track = detections[inew]
            if track.score < self.det_thresh:
                continue
            track.activate(self.kalman_filter, self.frame_id)
            activated_starcks.append(track)
        """ Step 5: Update state"""
        # 对于丢失目标的 track 来说，判断他丢失的帧数是不是超过了 buffer 缓冲帧数，超过就删除
        for track in self.lost_stracks:
            if self.frame_id - track.end_frame > self.max_time_lost:
                track.mark_removed()
                removed_stracks.append(track)

        # print('Ramained match {} s'.format(t4-t3))

        # 指上一帧匹配上的 track
        self.tracked_stracks = [t for t in self.tracked_stracks if t.state == TrackState.Tracked]
        # 加上这一帧新激活的 track（两次匹配到的 track，以及由 unconfirm 状态变为激活态的 track）
        self.tracked_stracks = joint_stracks(self.tracked_stracks, activated_starcks)
        # 加上丢帧目标重新被匹配的 track
        self.tracked_stracks = joint_stracks(self.tracked_stracks, refind_stracks)
        # self.lost_stracks 在经过这一帧的匹配之后如果被重新激活的话就将其移出列表
        self.lost_stracks = sub_stracks(self.lost_stracks, self.tracked_stracks)
        # 将这一帧丢失的 track 添加进列表
        self.lost_stracks.extend(lost_stracks)
        # self.lost_stracks 如果在缓冲帧数内一直没有被匹配上被 remove 的话也将其移出 lost_stracks 列表
        self.lost_stracks = sub_stracks(self.lost_stracks, self.removed_stracks)
        # 更新被移除的 track 列表
        self.removed_stracks.extend(removed_stracks)
        # 将这两段 track 中重合度高的部分给移除掉（暂时还不是特别理解为啥要这样）
        self.tracked_stracks, self.lost_stracks = remove_duplicate_stracks(self.tracked_stracks, self.lost_stracks)
        # get scores of lost tracks
        # 得到最终的结果，也就是成功追踪的 track 序列
        output_stracks = [track for track in self.tracked_stracks if track.is_activated]

        return output_stracks


# 将 tlista 和 tlistb 的 track 给合并成一个大的列表
def joint_stracks(tlista, tlistb):
    exists = {}
    res = []
    for t in tlista:
        exists[t.track_id] = 1
        res.append(t)
    for t in tlistb:
        tid = t.track_id
        if not exists.get(tid, 0):
            exists[tid] = 1
            res.append(t)
    return res

# 取两个 track 的不重合部分
def sub_stracks(tlista, tlistb):
    stracks = {}
    for t in tlista:
        stracks[t.track_id] = t
    for t in tlistb:
        tid = t.track_id
        if stracks.get(tid, 0):
            del stracks[tid]
    return list(stracks.values())


# 如果两段 track 离得很近的话，就要去掉一个
# 根据时间维度上出现的帧数多少来决定移除哪一边的 track
def remove_duplicate_stracks(stracksa, stracksb):
    pdist = matching.iou_distance(stracksa, stracksb)
    pairs = np.where(pdist < 0.15)
    dupa, dupb = list(), list()
    for p, q in zip(*pairs):
        timep = stracksa[p].frame_id - stracksa[p].start_frame
        timeq = stracksb[q].frame_id - stracksb[q].start_frame
        if timep > timeq:
            dupb.append(q)
        else:
            dupa.append(p)
    resa = [t for i, t in enumerate(stracksa) if not i in dupa]
    resb = [t for i, t in enumerate(stracksb) if not i in dupb]
    return resa, resb

```



## matching.py



只用到了这一个函数，用 IOU 作为匹配度量，计算 cost_matrix



```python
def iou_distance(atracks, btracks):
    """
    Compute cost based on IoU
    :type atracks: list[STrack]
    :type btracks: list[STrack]

    :rtype cost_matrix np.ndarray
    """

    if (len(atracks)>0 and isinstance(atracks[0], np.ndarray)) or (len(btracks) > 0 and isinstance(btracks[0], np.ndarray)):
        atlbrs = atracks
        btlbrs = btracks
    else:
      	# 注意这里调用的是 track.tlbr，是经过了卡尔曼 predict 之后的坐标！
        atlbrs = [track.tlbr for track in atracks]
        btlbrs = [track.tlbr for track in btracks]
    _ious = ious(atlbrs, btlbrs)
    cost_matrix = 1 - _ious

    return cost_matrix
```



## reference



https://zhuanlan.zhihu.com/p/90835266