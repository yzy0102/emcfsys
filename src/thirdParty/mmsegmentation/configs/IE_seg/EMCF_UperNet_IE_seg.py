# model settings
_base_ = [
    '../_base_/models/timm_vit_upernet-b16.py',
    "../_base_/datasets/cityscapes_768x768.py",
]

crop_size = (512, 512)
norm_cfg = dict(type='BN', requires_grad=True)
data_preprocessor = dict(
    type='SegDataPreProcessor',
    mean=[123., 123., 123.],
    std=[53., 53., 53.],
    bgr_to_rgb=True,
    size=crop_size,
    pad_val=0,
    seg_pad_val=255)

model = dict(
    type='EncoderDecoder',
    data_preprocessor=data_preprocessor,
    backbone=dict(
        type='TIMMBackbone',
        model_name = 'vit_base_patch16_224',
        frozenbackbone = False,
        pretrained = False, 
        img_size=(512, 512),
        patch_size=16,
        in_channels=3,
        out_indices=(2, 5, 8, 11),
        qkv_bias=True,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        ),
    neck=dict(
        type='MultiLevelNeck',
        in_channels=[768, 768, 768, 768],
        out_channels=768,
        scales=[4, 2, 1, 0.5]),
    decode_head=dict(
        type='UPerHead',
        in_channels=[768, 768, 768, 768],
        in_index=[0, 1, 2, 3],
        pool_scales=(1, 2, 3, 6),
        channels=512,
        dropout_ratio=0.1,
        num_classes=2,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=[
            dict(type='DiceLoss', use_sigmoid=False, loss_weight=2.),
            dict(type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.),
            dict(type='soft_dice_cldice', loss_weight=2.0, line_index=1)
            ]
        ),
    auxiliary_head=dict(
        type='FCNHead',
        in_channels=768,
        in_index=3,
        channels=256,
        num_convs=1,
        concat_input=False,
        dropout_ratio=0.1,
        num_classes=2,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=[
            dict(type='DiceLoss', use_sigmoid=False, loss_weight=2.),
            dict(type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.),
            dict(type='soft_dice_cldice', loss_weight=2.0, line_index=1)
            ]
            ),
    # model training and testing settings
    train_cfg=dict(),
    test_cfg=dict(mode='slide', crop_size=(512, 512) ,stride=(480, 480)))
    # test_cfg=dict(mode='whole')
    # ) 

load_from = None
work_dir = None


#---------------------dataset-------------------------
# dataset settings
dataset_type = 'LiverDataset'
data_root = None
crop_size = (512, 512)

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5),
    dict(type='Pad', size=crop_size),
    dict(type='PhotoMetricDistortion'),
    dict(type='PackSegInputs')
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    # dict(type='Resize', scale=(512, 512), keep_ratio=False),
    # add loading annotation after ``Resize`` because ground truth
    # does not need to do resize data transform
    dict(type='LoadAnnotations'),
    dict(type='PackSegInputs')
]
img_ratios = [0.5, 0.75, 1.0, 1.25, 1.5, 1.75]
tta_pipeline = [
    dict(type='LoadImageFromFile', backend_args=None),
    dict(
        type='TestTimeAug',
        transforms=[
            [
                dict(type='Resize', scale=(512, 512), keep_ratio=False),
            ],
            [
                dict(type='RandomFlip', prob=0.5, direction='horizontal'),
            ], [dict(type='LoadAnnotations')], [dict(type='PackSegInputs')]
        ])
]

train_dataloader = dict(
    batch_size=4,
    num_workers=10,
    persistent_workers=True,
    sampler=dict(type='InfiniteSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file = 'splits/train.txt',
        data_prefix=dict(
            img_path='image', seg_map_path='label'),
        pipeline=train_pipeline))

val_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file = 'splits/train.txt',
        data_prefix=dict(
            img_path='image', seg_map_path='label'),
        pipeline=test_pipeline))


val_evaluator = dict(type='IoUMetric', iou_metrics=['mIoU', 'mDice', 'mFscore'])
test_evaluator = val_evaluator


# -----------------Runtime settings-------------------
default_scope = 'mmseg'
env_cfg = dict(
    cudnn_benchmark=True,
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl'),
)
vis_backends = [dict(type='LocalVisBackend')]
visualizer = dict(
    type='SegLocalVisualizer', vis_backends=vis_backends, name='visualizer')


log_level = 'INFO'
load_from = None
resume = False

tta_model = dict(type='SegTTAModel')

# ------------------------------------------------
# optimizer
optimizer = dict(lr=0.0001, type='AdamW', weight_decay=0.01)
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



max_iters = 6000
val_interval = 1000

train_cfg = dict(
    type="IterBasedTrainLoop",
    max_iters=max_iters,
    val_interval=val_interval,
)
val_cfg = dict(type="ValLoop")
test_cfg = dict(type="TestLoop")
param_scheduler = [
    dict(
        type="LinearLR",
        start_factor=1e-3,
        begin=0,
        end=max_iters,
        by_epoch=False,
    ),
    dict(
        type="PolyLR",
        eta_min=1e-6,
        power=0.9,
        begin=val_interval,
        end=max_iters,
        by_epoch=False,
    ),
]

log_processor = dict(by_epoch=False)

default_hooks = dict(
    checkpoint=dict(
        type="CheckpointHook",
        by_epoch=False,
        interval=val_interval,
        max_keep_ckpts=1,
        save_best="mIoU",
    ),
    logger=dict(
        type="LoggerHook",
        interval=val_interval,
        log_metric_by_epoch=False,
    ),
    param_scheduler=dict(type="ParamSchedulerHook"),
    sampler_seed=dict(type="DistSamplerSeedHook"),
    timer=dict(type="IterTimerHook"),
    visualization=dict(type="SegVisualizationHook"),
)