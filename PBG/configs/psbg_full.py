# Full PSBG model: SM + PITE + MiT-B1 + boundary-aware decoder.
# Paths are relative to the directory from which the command is launched.

data_root = './data/gasvid'
num_frames = 3
temporal_stride = 5
dataset_type = 'NumericSharedWindowGasDatasetVideo'

model = dict(
    type='TimeEncoderDecoder',
    pretrained='./pretrained/mit_b1.pth',
    backbone=dict(
        type='tdmgas_original_shared_window_b1',
        num_frames=num_frames,
        drop_path_rate=0.2,
        memory_gate_scale=1.0,
        physics_cfg=dict(
            dt=1.0,
            velocity_scale=1.0,
            diffusion_scale=0.25,
            source_scale=0.1,
            conf_scale=2.0,
            gate_temporal_features=True)),
    decode_head=dict(
        type='GasPhysEdgeHead',
        in_channels=[64, 128, 320, 512],
        in_index=[0, 1, 2, 3],
        feature_strides=[4, 8, 16, 32],
        channels=128,
        dropout_ratio=0.3,
        num_classes=2,
        norm_cfg=dict(type='BN', requires_grad=True),
        align_corners=False,
        decoder_params=dict(embed_dim=256),
        loss_decode=[
            dict(
                type='OhemCrossEntropy',
                thres=0.9,
                min_kept=131072,
                loss_weight=1.0),
            dict(type='DiceLoss', loss_weight=1.0, class_weight=[1.0, 20.0])
        ],
        edge_channels=64,
        edge_gate_scale=1.0,
        edge_refine_scale=0.25,
        diffusion_iters=2,
        diffusion_rate=0.15,
        edge_loss_weight=0.1,
        edge_pos_weight=4.0),
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    to_rgb=True)
crop_size = (320, 256)

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', reduce_zero_label=False),
    dict(
        type='Resize', img_scale=(320, 256), ratio_range=(0.5, 2.0),
        process_clips=True),
    dict(type='RandomCrop_clips', crop_size=crop_size, cat_max_ratio=0.75),
    dict(type='RandomFlip_clips', prob=0.5),
    dict(type='PhotoMetricDistortion_clips'),
    dict(type='Normalize_clips', **img_norm_cfg),
    dict(type='Pad_clips', size=crop_size, pad_val=0, seg_pad_val=255),
    dict(type='DefaultFormatBundle_clips'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg'])
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(320, 256),
        flip=False,
        transforms=[
            dict(type='AlignedResize_clips', keep_ratio=True, size_divisor=32),
            dict(type='RandomFlip_clips'),
            dict(type='Normalize_clips', **img_norm_cfg),
            dict(type='ImageToTensor_clips', keys=['img']),
            dict(type='Collect', keys=['img'])
        ])
]

data = dict(
    samples_per_gpu=4,
    workers_per_gpu=0,
    dataloader_type='DataLoader',
    train=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='',
        split='train',
        pipeline=train_pipeline,
        num_frames=num_frames,
        temporal_stride=temporal_stride),
    val=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='',
        split='val',
        test_mode=True,
        pipeline=test_pipeline,
        num_frames=num_frames,
        temporal_stride=temporal_stride),
    test=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='',
        split='val',
        test_mode=True,
        pipeline=test_pipeline,
        num_frames=num_frames,
        temporal_stride=temporal_stride))

optimizer = dict(
    type='AdamW',
    lr=6e-5,
    betas=(0.9, 0.999),
    weight_decay=0.05,
    paramwise_cfg=dict(
        custom_keys=dict(
            pos_block=dict(decay_mult=0.0),
            norm=dict(decay_mult=0.0),
            head=dict(lr_mult=10.0))))
optimizer_config = dict()
lr_config = dict(
    policy='poly',
    warmup='linear',
    warmup_iters=1500,
    warmup_ratio=1e-6,
    power=1.0,
    min_lr=0.0,
    by_epoch=False)
runner = dict(type='IterBasedRunner', max_iters=160000)
checkpoint_config = dict(by_epoch=False, interval=10000)
evaluation = dict(interval=10000, metric='mIoU')
ema_config = dict(decay=0.999, start_iter=0)

log_config = dict(
    interval=50, hooks=[dict(type='TextLoggerHook', by_epoch=False)])
log_level = 'INFO'
workflow = [('train', 1)]
work_dir = './work_dirs/psbg_full'
load_from = None
resume_from = None
gpu_ids = [0]
cudnn_benchmark = True
find_unused_parameters = True
dist_params = dict(backend='nccl')
