crop_size = (
    512,
    512,
)
custom_imports = dict(
    imports=[
        'configs.register_models.runtime',
        'configs.register_models.plant_cell_dataset',
    ],
    allow_failed_imports=False,
)
data_preprocessor = dict(
    bgr_to_rgb=True,
    mean=[
        123.0,
        123.0,
        123.0,
    ],
    pad_val=0,
    seg_pad_val=255,
    size=(
        512,
        512,
    ),
    std=[
        53.0,
        53.0,
        53.0,
    ],
    type='SegDataPreProcessor')
data_root = '{{fileDirname}}/../../'
dataset_type = 'PlantCellDataset'
default_hooks = dict(
    checkpoint=dict(
        by_epoch=True,
        interval=2,
        max_keep_ckpts=2,
        save_best='mIoU',
        type='CheckpointHook'),
    logger=dict(interval=20, log_metric_by_epoch=True, type='LoggerHook'),
    param_scheduler=dict(type='ParamSchedulerHook'),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    timer=dict(type='IterTimerHook'),
    visualization=dict(type='SegVisualizationHook'))
default_scope = 'mmseg'
env_cfg = dict(
    cudnn_benchmark=True,   # 图像尺寸固定保持True；测试时图像大小多变改为False
    dist_cfg=dict(backend='nccl'),  # 单卡会自动跳过nccl，不用修改/删除
    mp_cfg=dict(mp_start_method='spawn', opencv_num_threads=0)
)
launcher = 'none'
find_unused_parameters = False
img_ratios = [
    0.5,
    0.75,
    1.0,
    1.25,
    1.5,
    1.75,
]
load_from = '{{fileDirname}}/../../models/DinoV3_EMCellFound_ViT_base.pth'
log_level = 'INFO'
log_processor = dict(by_epoch=True)
model = dict(
    auxiliary_head=dict(
        align_corners=False,
        channels=256,
        concat_input=False,
        dropout_ratio=0.1,
        in_channels=768,
        in_index=3,
        loss_decode=[
            dict(loss_weight=2.0, type='DiceLoss', use_sigmoid=False),
            dict(loss_weight=1.0, type='CrossEntropyLoss', use_sigmoid=False),
        ],
        norm_cfg=dict(requires_grad=True, type='BN'),
        num_classes=5,
        num_convs=1,
        type='FCNHead'),
    backbone=dict(
        depth=12,
        drop_path_rate=0.0,
        embed_dim=768,
        ffn_layer='swiglu64',
        img_size=512,
        in_chans=3,
        num_heads=12,
        out_indices=(
            2,
            5,
            8,
            11,
        ),
        patch_size=16,
        type='DINOv3Backbone'),
    data_preprocessor=dict(
        bgr_to_rgb=True,
        mean=[
            123.0,
            123.0,
            123.0,
        ],
        pad_val=0,
        seg_pad_val=255,
        size=(
            512,
            512,
        ),
        std=[
            53.0,
            53.0,
            53.0,
        ],
        type='SegDataPreProcessor'),
    decode_head=dict(
        align_corners=False,
        channels=512,
        dropout_ratio=0.1,
        in_channels=[
            768,
            768,
            768,
            768,
        ],
        in_index=[
            0,
            1,
            2,
            3,
        ],
        loss_decode=[
            dict(loss_weight=2.0, type='DiceLoss', use_sigmoid=False),
            dict(loss_weight=1.0, type='CrossEntropyLoss', use_sigmoid=False),
        ],
        norm_cfg=dict(requires_grad=True, type='BN'),
        num_classes=5,
        pool_scales=(
            1,
            2,
            3,
            6,
        ),
        type='UPerHead'),
    neck=dict(
        in_channels=[
            768,
            768,
            768,
            768,
        ],
        out_channels=768,
        scales=[
            4,
            2,
            1,
            0.5,
        ],
        type='MultiLevelNeck'),
    test_cfg=dict(mode='whole'),
    train_cfg=dict(),
    type='EncoderDecoder')
norm_cfg = dict(requires_grad=True, type='BN')
optim_wrapper = dict(
    clip_grad=dict(max_norm=1.0),
    optimizer=dict(lr=0.0001, type='AdamW', weight_decay=0.01),
    paramwise_cfg=dict(
        bias_decay_mult=0.0,
        custom_keys=dict(
            auxiliary_head=dict(lr_mult=10.0),
            backbone=dict(lr_mult=0.1),
            decode_head=dict(lr_mult=10.0)),
        norm_decay_mult=0.0),
    type='OptimWrapper')
optimizer = dict(lr=0.0001, type='AdamW', weight_decay=0.01)
param_scheduler = [
    dict(begin=0, by_epoch=True, end=10, start_factor=1e-06, type='LinearLR'),
    dict(
        begin=10,
        by_epoch=True,
        end=100,
        eta_min=1e-06,
        power=0.9,
        type='PolyLR'),
]
resume = False
test_cfg = dict(type='TestLoop')
test_dataloader = dict(
    batch_size=1,
    dataset=dict(
        ann_file='splits/test.txt',
        data_prefix=dict(img_path='image', seg_map_path='label'),
        data_root=data_root,
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(keep_ratio=False, scale=(
                512,
                512,
            ), type='Resize'),
            dict(type='LoadAnnotations'),
            dict(type='PackSegInputs'),
        ],
        type='PlantCellDataset'),
    num_workers=0,
    persistent_workers=False,
    sampler=dict(shuffle=False, type='DefaultSampler'))
test_evaluator = dict(
    iou_metrics=[
        'mIoU',
        'mDice',
        'mFscore',
    ], type='IoUMetric')
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(keep_ratio=False, scale=(
        512,
        512,
    ), type='Resize'),
    dict(type='LoadAnnotations'),
    dict(type='PackSegInputs'),
]
train_cfg = dict(max_epochs=100, type='EpochBasedTrainLoop', val_interval=2)
train_dataloader = dict(
    batch_size=8,
    dataset=dict(
        ann_file='splits/train.txt',
        data_prefix=dict(img_path='image', seg_map_path='label'),
        data_root=data_root,
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='LoadAnnotations'),
            dict(
                keep_ratio=True,
                ratio_range=(
                    0.5,
                    2.0,
                ),
                scale=(
                    768,
                    512,
                ),
                type='RandomResize'),
            dict(
                cat_max_ratio=0.75, crop_size=(
                    512,
                    512,
                ), type='RandomCrop'),
            dict(prob=0.5, type='RandomFlip'),
            dict(size=(
                512,
                512,
            ), type='Pad'),
            dict(type='PhotoMetricDistortion'),
            dict(type='PackSegInputs'),
        ],
        type='PlantCellDataset'),
    num_workers=0,
    persistent_workers=False,
    sampler=dict(shuffle=True, type='DefaultSampler'))
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(
        keep_ratio=True,
        ratio_range=(
            0.5,
            2.0,
        ),
        scale=(
            768,
            512,
        ),
        type='RandomResize'),
    dict(cat_max_ratio=0.75, crop_size=(
        512,
        512,
    ), type='RandomCrop'),
    dict(prob=0.5, type='RandomFlip'),
    dict(size=(
        512,
        512,
    ), type='Pad'),
    dict(type='PhotoMetricDistortion'),
    dict(type='PackSegInputs'),
]
tta_model = dict(type='SegTTAModel')
tta_pipeline = [
    dict(backend_args=None, type='LoadImageFromFile'),
    dict(
        transforms=[
            [
                dict(keep_ratio=False, scale=(
                    512,
                    512,
                ), type='Resize'),
            ],
            [
                dict(direction='horizontal', prob=0.5, type='RandomFlip'),
            ],
            [
                dict(type='LoadAnnotations'),
            ],
            [
                dict(type='PackSegInputs'),
            ],
        ],
        type='TestTimeAug'),
]
val_cfg = dict(type='ValLoop')
val_dataloader = dict(
    batch_size=1,
    dataset=dict(
        ann_file='splits/val.txt',
        data_prefix=dict(img_path='image', seg_map_path='label'),
        data_root=data_root,
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(keep_ratio=False, scale=(
                512,
                512,
            ), type='Resize'),
            dict(type='LoadAnnotations'),
            dict(type='PackSegInputs'),
        ],
        type='PlantCellDataset'),
    num_workers=0,
    persistent_workers=False,
    sampler=dict(shuffle=False, type='DefaultSampler'))
val_evaluator = dict(
    iou_metrics=[
        'mIoU',
        'mDice',
        'mFscore',
    ], type='IoUMetric')
vis_backends = [
    dict(type='LocalVisBackend'),
]
visualizer = dict(
    name='visualizer',
    type='SegLocalVisualizer',
    vis_backends=[
        dict(type='LocalVisBackend'),
    ])
work_dir = '{{fileDirname}}/../../save_logs/DinoV3_UperNet'
