import argparse
import sys
from pathlib import Path
import os
import os.path as osp

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import mmcv
import numpy as np
import torch
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import get_dist_info, init_dist, load_checkpoint
from mmcv.utils import DictAction
from PIL import Image

from mmseg.apis import multi_gpu_test, single_gpu_test
from mmseg.datasets import build_dataloader, build_dataset
from mmseg.models import build_segmentor

def parse_args():
    parser = argparse.ArgumentParser(
        description='mmseg test (and eval) a model')
    parser.add_argument('--config', default=str(Path(__file__).resolve().parents[1] / 'configs' / 'psbg_full.py'), help='test config file path')
    parser.add_argument('--checkpoint', required=True, help='full-model checkpoint file')
    parser.add_argument('--data-root', help='dataset root for evaluation')
    parser.add_argument('--split', choices=['val', 'test'], help='evaluation split')
    parser.add_argument('--out', default=None, help='output result file in pickle format')
    parser.add_argument(
        '--no-out', dest='out', action='store_const', const=None,
        help='disable saving prediction results as a pickle file')
    parser.add_argument(
        '--metrics-out', help='save evaluation metrics to a JSON file')
    parser.add_argument(
        '--eval',
        type=str,
        nargs='+',
        default=['mIoU'],
        help='evaluation metrics: mIoU or mDice; reports all four gas metrics')
    parser.add_argument('--show', action='store_true',default=None,  help='show results')
    parser.add_argument(
        '--show-dir',default=None, help='directory where painted images will be saved')
    parser.add_argument(
        '--mask-dir',default=None,
        help='directory where raw predicted mask png files will be saved')
    parser.add_argument(
        '--diff-dir', default=None,
        help='directory where label/prediction difference maps will be saved')
    parser.add_argument(
        '--gpu-collect',
        action='store_true',
        help='whether to use gpu to collect results.')
    parser.add_argument(
        '--tmpdir',
        help='tmp directory used for collecting results from multiple '
        'workers, available when gpu_collect is not specified')
    parser.add_argument(
        '--options', nargs='+', action=DictAction, help='custom options')
    parser.add_argument(
        '--eval-options',
        nargs='+',
        action=DictAction,
        help='custom options for evaluation')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)
    return args


def _load_seg_map(seg_map):
    if isinstance(seg_map, str):
        if seg_map.endswith('.npy'):
            return np.load(seg_map)
        return mmcv.imread(seg_map, flag='unchanged', backend='pillow')
    return seg_map


def _prepare_gt_seg_map(gt_seg_map, dataset):
    gt_seg_map = _load_seg_map(gt_seg_map).copy()
    if getattr(dataset, 'label_map', None) is not None:
        for old_id, new_id in dataset.label_map.items():
            gt_seg_map[gt_seg_map == old_id] = new_id
    if getattr(dataset, 'reduce_zero_label', False):
        gt_seg_map[gt_seg_map == 0] = 255
        gt_seg_map = gt_seg_map - 1
        gt_seg_map[gt_seg_map == 254] = 255
    return gt_seg_map


def _get_diff_filename(dataset, idx):
    if hasattr(dataset, 'img_all'):
        video, img_name = dataset.img_all[idx]
        rel_path = osp.join(video, img_name)
    else:
        rel_path = dataset.img_infos[idx]['filename']
    rel_path = osp.splitext(rel_path)[0] + '_diff.png'
    return rel_path


def save_diff_maps(results, dataset, out_dir):
    assert hasattr(dataset, 'get_gt_seg_maps'), \
        f'{type(dataset)} does not support loading ground-truth masks'
    assert len(results) == len(dataset), \
        f'Dataset and results have different sizes: {len(dataset)} v.s. {len(results)}'

    gt_seg_maps = dataset.get_gt_seg_maps(efficient_test=True)
    ignore_index = getattr(dataset, 'ignore_index', 255)

    for idx, result in enumerate(results):
        pred = _load_seg_map(result)
        label = _prepare_gt_seg_map(gt_seg_maps[idx], dataset)
        if pred.shape != label.shape:
            pred = mmcv.imresize(
                pred, (label.shape[1], label.shape[0]), interpolation='nearest')

        valid = label != ignore_index
        correct = (pred == label) & valid
        false_positive = (pred > 0) & (label == 0) & valid
        false_negative = (pred == 0) & (label > 0) & valid
        other_error = (~correct) & valid & (~false_positive) & (~false_negative)

        diff = np.zeros((*label.shape, 3), dtype=np.uint8)
        diff[~valid] = (128, 128, 128)
        diff[false_positive] = (255, 0, 0)
        diff[false_negative] = (0, 0, 255)
        diff[other_error] = (255, 255, 0)

        out_file = osp.join(out_dir, _get_diff_filename(dataset, idx))
        mmcv.mkdir_or_exist(osp.dirname(out_file))
        Image.fromarray(diff, mode='RGB').save(out_file)


def main():
    args = parse_args()

    assert args.out or args.eval or args.show \
        or args.show_dir or args.mask_dir or args.diff_dir, \
        ('Please specify at least one operation (save/eval/format/show the '
         'results / save the results) with the argument "--out", "--eval"'
         ', "--format-only", "--show", "--show-dir", "--mask-dir" or '
         '"--diff-dir"')

    if 'None' in args.eval:
        args.eval = None
    if args.metrics_out and not args.eval:
        raise ValueError('--metrics-out requires --eval')
    if args.out is not None and not args.out.endswith(('.pkl', '.pickle')):
        raise ValueError('The output file must be a pkl file.')

    cfg = mmcv.Config.fromfile(args.config)
    if args.options is not None:
        cfg.merge_from_dict(args.options)
    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True
    if args.data_root is not None:
        cfg.data.test.data_root = args.data_root
    if args.split is not None:
        cfg.data.test.split = args.split

    cfg.model.pretrained = None
    cfg.data.test.test_mode = True

    # init distributed env first, since logger depends on the dist info.
    if args.launcher == 'none':
        distributed = False
    else:
        distributed = True
        init_dist(args.launcher, **cfg.dist_params)

    # build the dataloader
    # TODO: support multiple images per gpu (only minor changes are needed)
    dataset = build_dataset(cfg.data.test)
    data_loader = build_dataloader(
        dataset,
        samples_per_gpu=1,
        workers_per_gpu=cfg.data.workers_per_gpu,
        dist=distributed,
        shuffle=False)

    # build the model and load checkpoint
    cfg.model.train_cfg = None
    model = build_segmentor(cfg.model, test_cfg=cfg.get('test_cfg'))
    checkpoint = load_checkpoint(model, args.checkpoint, map_location='cpu')
    model.CLASSES = checkpoint['meta']['CLASSES']
    model.PALETTE = checkpoint['meta']['PALETTE']

    efficient_test = False
    if args.eval_options is not None:
        efficient_test = args.eval_options.get('efficient_test', False)

    if not distributed:
        model = MMDataParallel(model, device_ids=[0])
        outputs = single_gpu_test(model, data_loader, args.show, args.show_dir,
                                  efficient_test)
    else:
        model = MMDistributedDataParallel(
            model.cuda(),
            device_ids=[torch.cuda.current_device()],
            broadcast_buffers=False)
        outputs = multi_gpu_test(model, data_loader, args.tmpdir,
                                 args.gpu_collect, efficient_test)

    rank, _ = get_dist_info()
    if rank == 0:
        if args.out:
            print(f'\nwriting results to {args.out}')
            mmcv.mkdir_or_exist(osp.dirname(osp.abspath(args.out)))
            mmcv.dump(outputs, args.out)
        if args.mask_dir:
            print(f'\nwriting predicted masks to {args.mask_dir}')
            dataset.format_results(outputs, save_path=args.mask_dir)
        if args.diff_dir:
            print(f'\nwriting label/prediction difference maps to {args.diff_dir}')
            save_diff_maps(outputs, dataset, args.diff_dir)
        kwargs = {} if args.eval_options is None else args.eval_options
        if args.eval:
            metrics = dataset.evaluate(outputs, args.eval, **kwargs)
            if args.metrics_out:
                mmcv.mkdir_or_exist(osp.dirname(osp.abspath(args.metrics_out)))
                mmcv.dump(metrics, args.metrics_out)


if __name__ == '__main__':
    main()
