auto_scale_lr = dict(base_batch_size=16, enable=False)
backend_args = None
base_lr = 0.001
max_epochs = 400
custom_hooks = [
    dict(
        ema_type='ExpMomentumEMA',
        momentum=0.0002,
        priority=49,
        type='EMAHook',
        update_buffers=True),
    dict(
        switch_epoch=280,
        switch_pipeline=[
            dict(backend_args=None, type='LoadImageFromFile'),
            dict(
                poly2mask=False,
                type='LoadAnnotations',
                with_bbox=True,
                with_mask=True),
            dict(
                allow_negative_crop=True,
                crop_size=(
                    1024,
                    1024,
                ),
                recompute_bbox=True,
                type='RandomCrop'),
            dict(
                keep_ratio=True,
                ratio_range=(
                    0.1,
                    2.0,
                ),
                scale=(
                    512,
                    512,
                ),
                type='RandomResize'),
            dict(
                allow_negative_crop=True,
                crop_size=(
                    512,
                    512,
                ),
                recompute_bbox=True,
                type='RandomCrop'),
            dict(min_gt_bbox_wh=(
                1,
                1,
            ), type='FilterAnnotations'),
            dict(type='YOLOXHSVRandomAug'),
            dict(prob=0.5, type='RandomFlip'),
            dict(
                pad_val=dict(img=(
                    114,
                    114,
                    114,
                )),
                size=(
                    512,
                    512,
                ),
                type='Pad'),
            dict(type='PackDetInputs'),
        ],
        type='PipelineSwitchHook'),
]

dataset_type = 'CocoDataset'
default_hooks = dict(
    checkpoint=dict(interval=2, max_keep_ckpts=3, save_best = "auto", type='CheckpointHook'),
    logger=dict(interval=10, type='LoggerHook'),
    param_scheduler=dict(type='ParamSchedulerHook'),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    timer=dict(type='IterTimerHook'),
    visualization=dict(type='DetVisualizationHook'))
default_scope = 'mmdet'
env_cfg = dict(
    cudnn_benchmark=False,
    dist_cfg=dict(backend='nccl'),
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0))
img_scales = [
    (
        512,
        512,
    ),
    (
        320,
        320,
    ),
    (
        960,
        960,
    ),
]
interval = 2
load_from = None
log_level = 'INFO'
log_processor = dict(by_epoch=True, type='LogProcessor', window_size=50)

metainfo = dict(
    classes=('Mitochondria', ), palette=[
        (
            220,
            20,
            60,
        ),
    ])
model = dict(
    backbone=dict(
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        drop_rate=0.0,
        frozenbackbone=False,
        img_size=(
            512,
            512,
        ),
        in_channels=3,
        model_name='vit_base_patch16_224',
        out_indices=(
            2,
            5,
            8,
            11,
        ),
        patch_size=16,
        pretrained=False,
        qkv_bias=True,
        type='Multi_ViT'),
    neck=dict(
        act_cfg=dict(inplace=True, type='SiLU'),
        expand_ratio=0.5,
        in_channels=[
            768,
            768,
            768,
        ],
        norm_cfg=dict(type='SyncBN'),
        num_csp_blocks=3,
        out_channels=256,
        type='CSPNeXtPAFPN'),
    bbox_head=dict(
        act_cfg=dict(inplace=True, type='SiLU'),
        anchor_generator=dict(
            offset=0, strides=[
                8,
                16,
                32,
            ], type='MlvlPointGenerator'),
        bbox_coder=dict(type='DistancePointBBoxCoder'),
        feat_channels=256,
        in_channels=256,
        loss_bbox=dict(loss_weight=2.0, type='GIoULoss'),
        loss_cls=dict(
            beta=2.0,
            loss_weight=1.0,
            type='QualityFocalLoss',
            use_sigmoid=True),
        loss_mask=dict(
            eps=5e-06, loss_weight=2.0, reduction='mean', type='DiceLoss'),
        norm_cfg=dict(requires_grad=True, type='SyncBN'),
        num_classes=1,
        pred_kernel_size=1,
        share_conv=True,
        stacked_convs=2,
        type='RTMDetInsSepBNHead'),
    data_preprocessor=dict(
        batch_augments=None,
        bgr_to_rgb=False,
        mean=[
            123.,
            123.,
            123.,
        ],
        std=[
            53.,
            53.,
            53.,
        ],
        type='DetDataPreprocessor'),

    test_cfg=dict(
        mask_thr_binary=0.5,
        max_per_img=100,
        min_bbox_size=0,
        nms=dict(iou_threshold=0.6, type='soft_nms'),
        nms_pre=1000,
        score_thr=0.05),
    train_cfg=dict(
        allowed_border=-1,
        assigner=dict(topk=13, type='DynamicSoftLabelAssigner'),
        debug=False,
        pos_weight=-1),
    type='RTMDet')
optim_wrapper = dict(
    clip_grad=dict(max_norm=1.0),
    optimizer=dict(lr=0.001, type='AdamW', weight_decay=0.01),
    paramwise_cfg=dict(
        bias_decay_mult=0.0,
        bypass_duplicate=True,
        custom_keys=dict(
            backbone=dict(lr_mult=0.01)),
        norm_decay_mult=0.0),
    type='OptimWrapper')
param_scheduler = [
    dict(
        begin=0, by_epoch=True, end=max_epochs//4, start_factor=1e-05,
        type='LinearLR'),
]
resume = False
stage2_num_epochs = 20
test_cfg = dict(type='TestLoop')
test_dataloader = dict(
    batch_size=5,
    dataset=dict(
        ann_file='test.json',
        backend_args=None,
        data_prefix=dict(img='image/'),
        data_root='',
        metainfo=dict(classes=('Mitochondria', ), palette=[
            (
                220,
                20,
                60,
            ),
        ]),
        pipeline=[
            dict(backend_args=None, type='LoadImageFromFile'),
            dict(keep_ratio=True, scale=(
                512,
                512,
            ), type='Resize'),
            dict(
                pad_val=dict(img=(
                    114,
                    114,
                    114,
                )),
                size=(
                    512,
                    512,
                ),
                type='Pad'),
            dict(type='LoadAnnotations', with_bbox=True),
            dict(
                meta_keys=(
                    'img_id',
                    'img_path',
                    'ori_shape',
                    'img_shape',
                    'scale_factor',
                ),
                type='PackDetInputs'),
        ],
        test_mode=True,
        type='CocoDataset'),
    drop_last=False,
    num_workers=10,
    persistent_workers=True,
    sampler=dict(shuffle=False, type='DefaultSampler'))
test_evaluator = dict(
    ann_file='test.json',
    backend_args=None,
    format_only=False,
    metric=[
        'bbox',
        'segm',
    ],
    proposal_nums=(
        100,
        1,
        10,
    ),
    type='CocoMetric')
test_pipeline = [
    dict(backend_args=None, type='LoadImageFromFile'),
    dict(keep_ratio=True, scale=(
        512,
        512,
    ), type='Resize'),
    dict(pad_val=dict(img=(
        114,
        114,
        114,
    )), size=(
        512,
        512,
    ), type='Pad'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        meta_keys=(
            'img_id',
            'img_path',
            'ori_shape',
            'img_shape',
            'scale_factor',
        ),
        type='PackDetInputs'),
]
train_cfg = dict(
    dynamic_intervals=[
        (
            280,
            1,
        ),
    ],
    max_epochs=max_epochs,
    type='EpochBasedTrainLoop',
    val_interval=2)

train_cfg = dict(max_epochs=400, type='EpochBasedTrainLoop', val_interval=20)
train_dataloader = dict(
    batch_size=8,
    dataset=dict(
        ann_file=
        'train.json',
        data_prefix=dict(img=''),
        data_root="",
        filter_cfg=dict(filter_empty_gt=True, min_size=1),
        metainfo=dict(
            classes=('Mitochondria', ), palette=[
                (
                    46,
                    139,
                    87,
                ),
            ]),
        pipeline=[
            dict(backend_args=None, type='LoadImageFromFile'),
            dict(
                poly2mask=False,
                type='LoadAnnotations',
                with_bbox=True,
                with_mask=True),
            dict(
                keep_ratio=True,
                ratio_range=(
                    0.8,
                    1.2,
                ),
                scale=(
                    1285,
                    1285,
                ),
                type='RandomResize'),
            dict(
                allow_negative_crop=True,
                crop_size=(
                    512,
                    512,
                ),
                recompute_bbox=True,
                type='RandomCrop'),
            dict(min_gt_bbox_wh=(
                1,
                1,
            ), type='FilterAnnotations'),
            dict(type='YOLOXHSVRandomAug'),
            dict(prob=0.5, type='RandomFlip'),
            dict(
                pad_val=dict(img=(
                    114,
                    114,
                    114,
                )),
                size=(
                    512,
                    512,
                ),
                type='Pad'),
            dict(type='PackDetInputs'),
        ],
        type='CocoDataset'))


train_pipeline = [
    dict(backend_args=None, type='LoadImageFromFile'),
    dict(
        poly2mask=False,
        type='LoadAnnotations',
        with_bbox=True,
        with_mask=True),
    dict(
        keep_ratio=True,
        ratio_range=(0.5, 1.5), 
        scale=(1024, 1024), 
        type='RandomResize'),
    dict(
        allow_negative_crop=False, 
        crop_size=(512, 512),
        recompute_bbox=True,
        type='RandomCrop'),
    dict(prob=0.5, direction='horizontal', type='RandomFlip'),
    dict(prob=0.5, direction='vertical', type='RandomFlip'),
    dict(prob=0.5, type='RandomRotate90'),
    dict(
        pad_val=dict(img=(114, 114, 114)),
        size=(512, 512),
        type='Pad'),
    dict(min_gt_bbox_wh=(2, 2), type='FilterAnnotations'),
    dict(type='PackDetInputs'),
]

# stage2
train_pipeline_stage2 = [
    dict(backend_args=None, type='LoadImageFromFile'),
    dict(
        poly2mask=False,
        type='LoadAnnotations',
        with_bbox=True,
        with_mask=True),
    dict(
        keep_ratio=True,
        ratio_range=(0.8, 1.2),  # finetune阶段进一步收缩缩放，减少形变
        scale=(512, 512),
        type='RandomResize'),
    dict(
        allow_negative_crop=False,
        crop_size=(512, 512),
        recompute_bbox=True,
        type='RandomCrop'),
    dict(min_gt_bbox_wh=(2, 2), type='FilterAnnotations'),
    # finetune依旧保留翻转+90旋转
    dict(prob=0.5, direction='horizontal', type='RandomFlip'),
    dict(prob=0.5, direction='vertical', type='RandomFlip'),
    dict(prob=0.5, type='RandomRotate90'),
    dict(
        pad_val=dict(img=(114, 114, 114)),
        size=(512, 512),
        type='Pad'),
    dict(type='PackDetInputs'),
]
tta_model = dict(
    tta_cfg=dict(max_per_img=100, nms=dict(iou_threshold=0.6, type='soft_nms')),
    type='DetTTAModel')
tta_pipeline = [
    dict(backend_args=None, type='LoadImageFromFile'),
    dict(
        transforms=[
            [
                dict(keep_ratio=True, scale=(
                    512,
                    512,
                ), type='Resize'),
                dict(keep_ratio=True, scale=(
                    320,
                    320,
                ), type='Resize'),
                dict(keep_ratio=True, scale=(
                    960,
                    960,
                ), type='Resize'),
            ],
            [
                dict(prob=1.0, type='RandomFlip'),
                dict(prob=0.0, type='RandomFlip'),
            ],
            [
                dict(
                    pad_val=dict(img=(
                        114,
                        114,
                        114,
                    )),
                    size=(
                        960,
                        960,
                    ),
                    type='Pad'),
            ],
            [
                dict(type='LoadAnnotations', with_bbox=True),
            ],
            [
                dict(
                    meta_keys=(
                        'img_id',
                        'img_path',
                        'ori_shape',
                        'img_shape',
                        'scale_factor',
                        'flip',
                        'flip_direction',
                    ),
                    type='PackDetInputs'),
            ],
        ],
        type='TestTimeAug'),
]
val_cfg = dict(type='ValLoop')
val_dataloader = dict(
    batch_size=5,
    dataset=dict(
        ann_file='val.json',
        backend_args=None,
        data_prefix=dict(img='image/'),
        data_root='',
        metainfo=dict(classes=('Mitochondria', ), palette=[
            (
                220,
                20,
                60,
            ),
        ]),
        pipeline=[
            dict(backend_args=None, type='LoadImageFromFile'),
            dict(keep_ratio=True, scale=(
                512,
                512,
            ), type='Resize'),
            dict(
                pad_val=dict(img=(
                    114,
                    114,
                    114,
                )),
                size=(
                    512,
                    512,
                ),
                type='Pad'),
            dict(type='LoadAnnotations', with_bbox=True),
            dict(
                meta_keys=(
                    'img_id',
                    'img_path',
                    'ori_shape',
                    'img_shape',
                    'scale_factor',
                ),
                type='PackDetInputs'),
        ],
        test_mode=True,
        type='CocoDataset'),
    drop_last=False,
    num_workers=10,
    persistent_workers=True,
    sampler=dict(shuffle=False, type='DefaultSampler'))
val_evaluator = dict(
    ann_file='val.json',
    backend_args=None,
    format_only=False,
    metric=[
        'bbox',
        'segm',
    ],
    proposal_nums=(
        100,
        1,
        10,
    ),
    type='CocoMetric')
vis_backends = [
    dict(type='LocalVisBackend'),
]
visualizer = dict(
    name='visualizer',
    type='DetLocalVisualizer',
    vis_backends=[
        dict(type='LocalVisBackend'),
    ])
work_dir = './tutorial_exps'

# EMCFsys COCO dataset override. The dataset root contains image/ and the
# train.json, val.json, and test.json annotation files.
data_root = '{{fileDirname}}/../../../datasets_temp/MitoInstanceSegDataset'
dataset_type = 'CocoDataset'
launcher = 'none'
for _split_name, _loader in (('train', train_dataloader),
                             ('val', val_dataloader),
                             ('test', test_dataloader)):
    _dataset = _loader['dataset']
    _dataset['type'] = dataset_type
    _dataset['ann_file'] = f'{_split_name}.json'
    _dataset['data_root'] = data_root
    _dataset['data_prefix'] = dict(img='image/')
    _dataset['metainfo'] = dict(
        classes=('Mitochondria',), palette=[(220, 20, 60)])
    _loader['num_workers'] = 0
    _loader['persistent_workers'] = False
val_evaluator['ann_file'] = f'{data_root}/val.json'
test_evaluator['ann_file'] = f'{data_root}/test.json'
work_dir = '{{fileDirname}}/../../../save_logs/ViT_RTM'

