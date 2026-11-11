# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import os
from .pipelines import Compose
import mmcv
from mmcv.utils import print_log
from mmseg.core import eval_metrics
from functools import reduce
from terminaltables import AsciiTable

class GasDataset_video(Dataset):

    CLASSES = ('_background_', 'gas')

    PALETTE = [[0, 0, 0], [128, 0, 0]]

    def __init__(self,
                 pipeline,
                 img_dir,
                 img_suffix='.jpg',
                 ann_dir=None,
                 seg_map_suffix='.png',
                 split=None,
                 data_root=None,
                 test_mode=False,
                 ignore_index=255,
                 reduce_zero_label=False,
                 classes=None,
                 palette=None,
                 time_dilation=4,
                 temporal_stride=1,
                 clipnum=None):
        # self.pipeline = Compose(pipeline)
        if split == 'train':
            self.pipeline_load = Compose(pipeline[:2])
            self.pipeline_process = Compose(pipeline[2:])
        else:
            self.pipeline_load = Compose(pipeline[:1])
            self.pipeline_process = Compose(pipeline[1:])

        self.img_dir = ''
        self.img_suffix = img_suffix
        self.ann_dir = ''
        self.seg_map_suffix = seg_map_suffix
        self.split = split
        self.data_root = data_root
        # print("here1: ", test_mode)
        self.test_mode = test_mode
        self.ignore_index = ignore_index
        self.reduce_zero_label = reduce_zero_label
        self.label_map = None
        self.CLASSES, self.PALETTE = self.get_classes_and_palette(
            classes, palette)

        self.time_dilation = time_dilation
        self.temporal_stride = temporal_stride
        self.clipnum = clipnum
        self.null_count = 0

        with open(os.path.join(self.data_root, self.split + '.txt')) as f:
            lines = f.readlines()
            self.videolists = [line[:-1] for line in lines]

        self.imgdic = {}
        self.img_all = []
        for video in self.videolists:
            v_path = os.path.join(self.data_root, 'data', video, 'origin')
            imglist = sorted(os.listdir(v_path))
            self.imgdic[video] = imglist
            self.img_all = self.img_all + [[video, img] for img in imglist]



    def __len__(self):
        """Total number of samples of data."""
        # return len(self.img_infos)
        if self.split == 'train':
            return len(self.videolists)
        else:
            return len(self.img_all)
        # return len(self.img_all)

    def __getitem__(self, idx):
        """Get training/test data after pipeline.

        Args:
            idx (int): Index of data.

        Returns:
            dict: Training/test data (with annotation if `test_mode` is set
                False).
        """
        # print("here: ", self.test_mode)
        if self.test_mode:
            return self.prepare_test_img(idx)
        else:
            return self.prepare_train_img(idx)



    def pre_pipeline(self, results, img_dir, ann_dir):
        """Prepare results dict for pipeline."""
        results['seg_fields'] = []
        results['img_prefix'] = img_dir
        results['seg_prefix'] = ann_dir
        if self.custom_classes:
            results['label_map'] = self.label_map








    def format_results(self, results, save_path=None, **kwargs):
        """Place holder to format result to dataset specific output."""
        # pass
        ## changed by guosun
        assert len(results) == len(self.img_all)
        palette_list = []
        for kk in self.PALETTE:
            palette_list = palette_list + kk

        for ii in range(len(results)):
            result = results[ii]
            video_imgname = self.img_all[ii]
            video, img_name = video_imgname[0], video_imgname[1]
            if isinstance(result, str):
                result = np.load(result)
            save_path_ii = os.path.join(save_path, 'result_submission', video,
                                        img_name.replace(self.img_suffix, self.seg_map_suffix))
            save_path_directory = os.path.dirname(save_path_ii)
            if not os.path.exists(save_path_directory):
                os.makedirs(save_path_directory)
            res = Image.fromarray(result.astype(np.uint8), mode='P')
            res.putpalette(palette_list)
            res.save(save_path_ii)

    def get_gt_seg_maps(self, efficient_test=False):
        """Get ground truth segmentation maps for evaluation."""

        gt_seg_maps = []
        for video_imgname in self.img_all:
            video, img_name = video_imgname[0], video_imgname[1]
            seg_map = img_name.replace(self.img_suffix, self.seg_map_suffix)
            ann_dir = os.path.join(self.data_root, 'data', video, 'mask/')

            seg_map = osp.join(ann_dir, seg_map)
            if efficient_test:
                gt_seg_map = seg_map
            else:
                gt_seg_map = mmcv.imread(
                    seg_map, flag='unchanged', backend='pillow')
            gt_seg_maps.append(gt_seg_map)

        return gt_seg_maps

    def get_classes_and_palette(self, classes=None, palette=None):
        """Get class names of current dataset.

        Args:
            classes (Sequence[str] | str | None): If classes is None, use
                default CLASSES defined by builtin dataset. If classes is a
                string, take it as a file name. The file contains the name of
                classes where each line contains one class name. If classes is
                a tuple or list, override the CLASSES defined by the dataset.
            palette (Sequence[Sequence[int]]] | np.ndarray | None):
                The palette of segmentation map. If None is given, random
                palette will be generated. Default: None
        """
        if classes is None:
            self.custom_classes = False
            return self.CLASSES, self.PALETTE

        self.custom_classes = True
        if isinstance(classes, str):
            # take it as a file path
            class_names = mmcv.list_from_file(classes)
        elif isinstance(classes, (tuple, list)):
            class_names = classes
        else:
            raise ValueError(f'Unsupported type {type(classes)} of classes.')

        if self.CLASSES:
            if not set(classes).issubset(self.CLASSES):
                raise ValueError('classes is not a subset of CLASSES.')

            # dictionary, its keys are the old label ids and its values
            # are the new label ids.
            # used for changing pixel labels in load_annotations.
            self.label_map = {}
            for i, c in enumerate(self.CLASSES):
                if c not in class_names:
                    self.label_map[i] = -1
                else:
                    self.label_map[i] = classes.index(c)

        palette = self.get_palette_for_custom_classes(class_names, palette)

        return class_names, palette

    def get_palette_for_custom_classes(self, class_names, palette=None):

        if self.label_map is not None:
            # return subset of palette
            palette = []
            for old_id, new_id in sorted(
                    self.label_map.items(), key=lambda x: x[1]):
                if new_id != -1:
                    palette.append(self.PALETTE[old_id])
            palette = type(self.PALETTE)(palette)

        elif palette is None:
            if self.PALETTE is None:
                palette = np.random.randint(0, 255, size=(len(class_names), 3))
            else:
                palette = self.PALETTE

        return palette

    def evaluate(self,
                 results,
                 metric='mIoU',
                 logger=None,
                 efficient_test=False,
                 **kwargs):
        """Evaluate the dataset."""

        # 1. 获取 Ground Truth
        eval_results = {}
        gt_seg_maps = self.get_gt_seg_maps(efficient_test)
        if self.CLASSES is None:
            num_classes = len(
                reduce(np.union1d, [np.unique(_) for _ in gt_seg_maps]))
        else:
            num_classes = len(self.CLASSES)

        # 2. 调用我们刚刚修改过的 eval_metrics
        # 它现在返回4个项: IoU, Recall, Precision, Dice
        ret_metrics = eval_metrics(
            results,
            gt_seg_maps,
            num_classes,
            self.ignore_index,
            metric,  # 这里的参数已经不起决定作用了，底层全都会算
            label_map=self.label_map,
            reduce_zero_label=self.reduce_zero_label)

        # 3. 准备表格数据
        # 定义表头
        class_table_data = [['Class', 'IoU', 'Recall', 'Precision', 'Dice']]

        if self.CLASSES is None:
            class_names = tuple(range(num_classes))
        else:
            class_names = self.CLASSES

        # 将数据乘以100并保留2位小数
        ret_metrics_round = [
            np.round(ret_metric * 100, 2) for ret_metric in ret_metrics
        ]

        # 4. 填充每一类的数据
        # ret_metrics_round 的顺序是 [IoU, Recall, Precision, Dice]
        for i in range(num_classes):
            class_table_data.append([class_names[i]] +
                                    [m[i] for m in ret_metrics_round])

        # 5. 计算全局平均 (Summary)
        summary_table_data = [['Scope', 'mIoU', 'mRecall', 'mPrecision', 'mDice']]

        # 计算每一列的平均值
        ret_metrics_mean = [
            np.round(np.nanmean(ret_metric) * 100, 2)
            for ret_metric in ret_metrics
        ]

        summary_table_data.append(['global'] + ret_metrics_mean)

        # 6. 打印输出
        print_log('per class results:', logger)
        table = AsciiTable(class_table_data)
        print_log('\n' + table.table, logger=logger)

        print_log('Summary:', logger)
        table = AsciiTable(summary_table_data)
        print_log('\n' + table.table, logger=logger)

        # 7. 返回给日志记录器 (Hook)
        eval_results['mIoU'] = ret_metrics_mean[0]
        eval_results['mRecall'] = ret_metrics_mean[1]
        eval_results['mPrecision'] = ret_metrics_mean[2]
        eval_results['mDice'] = ret_metrics_mean[3]

        # 清理临时文件
        if mmcv.is_list_of(results, str):
            for file_name in results:
                os.remove(file_name)

        return eval_results
