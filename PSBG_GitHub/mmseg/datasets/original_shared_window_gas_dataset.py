import os.path as osp

import numpy as np
import torch
from mmcv.parallel import DataContainer as DC

from .builder import DATASETS
from .gasDataset import GasDataset_video


@DATASETS.register_module()
class OriginalSharedWindowGasDatasetVideo(GasDataset_video):
    """Original video sampler with a shared, fixed-stride historical window.

    Training retains one dataset entry per video and randomly samples a
    complete window when available. Validation/test visit every target frame.
    Short histories are never resampled at a smaller stride or filled with
    future frames: they run only the current image with SPTE disabled.
    """

    def __init__(self, num_frames=3, temporal_stride=5, **kwargs):
        if not isinstance(num_frames, int) or num_frames < 1:
            raise ValueError('num_frames must be a positive integer')
        if not isinstance(temporal_stride, int) or temporal_stride < 1:
            raise ValueError('temporal_stride must be a positive integer')
        super().__init__(time_dilation=num_frames,
                         temporal_stride=temporal_stride, **kwargs)
        self.num_frames = num_frames

    def _window_indices(self, target):
        span = (self.num_frames - 1) * self.temporal_stride
        if self.num_frames == 1 or target < span:
            return [target]
        return list(range(target - span, target + 1, self.temporal_stride))

    def _load_frame(self, video, name, with_ann):
        info = dict(filename=name)
        results = dict(img_info=info)
        if with_ann:
            annotation = dict(seg_map=name.replace(
                self.img_suffix, self.seg_map_suffix))
            info['ann'] = annotation
            results['ann_info'] = annotation
        self.pre_pipeline(
            results, osp.join(self.data_root, 'data', video, 'origin'),
            osp.join(self.data_root, 'data', video, 'mask'))
        if with_ann:
            return self.pipeline_load(results)
        return self.pipeline_load.transforms[0](results)

    def _prepare_window(self, video, target, with_ann):
        indices = self._window_indices(target)
        frames = []
        for index in indices:
            last = self._load_frame(
                video, self.imgdic[video][index], with_ann and index == target)
            frames.append(last['img'])
        filename = last['ori_filename']
        if not with_ann:
            filename = osp.join(video, filename)
        results = dict(
            img_info=last['img_info'],
            seg_fields=last.get('seg_fields', []),
            img_prefix=last['img_prefix'], filename=last['filename'],
            ori_filename=filename, img=frames,
            img_shape=last['img_shape'], ori_shape=last['ori_shape'],
            pad_shape=last['pad_shape'], scale_factor=last['scale_factor'],
            img_norm_cfg=last['img_norm_cfg'], scene_id=video)
        for key in ('ann_info', 'seg_prefix'):
            if key in last:
                results[key] = last[key]
        if with_ann:
            results['gt_semantic_seg'] = [last['gt_semantic_seg']]
        data = self.pipeline_process(results)
        if with_ann:
            if len(indices) < self.num_frames:
                image = data['img'].data
                padded = image.new_zeros((self.num_frames, *image.shape[1:]))
                padded[-1] = image[-1]
                data['img'] = DC(padded, stack=True)
            data['spte_enabled'] = torch.tensor(
                self.num_frames > 1 and len(indices) == self.num_frames)
        return data

    def prepare_train_img(self, idx):
        video = self.videolists[idx]
        count = len(self.imgdic[video])
        span = (self.num_frames - 1) * self.temporal_stride
        start = span if count > span else 0
        target = np.random.randint(start, count)
        return self._prepare_window(video, target, with_ann=True)

    def prepare_test_img(self, idx):
        video, name = self.img_all[idx]
        target = self.imgdic[video].index(name)
        return self._prepare_window(video, target, with_ann=False)
