_data_root = ''
_train_ann = 'train.json'
custom_imports = dict(
    imports=[
        'configs.register_models.runtime',
        'configs.InstanceSegmentation.mmdet_register_models.ViT_backbone',
        'configs.InstanceSegmentation.mmdet_register_models.ViT_Adapter_backbone',
        'configs.InstanceSegmentation.mmdet_metrics',
    ],
    allow_failed_imports=False,
)

_train_pipeline = [
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
    dict(pad_val=dict(img=(
        114,
        114,
        114,
    )), size=(
        512,
        512,
    ), type='Pad'),
    dict(type='PackDetInputs'),
]
_val_ann = 'val.json'
_val_pipeline = [
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
    dict(
        poly2mask=False,
        type='LoadAnnotations',
        with_bbox=True,
        with_mask=True),
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
auto_scale_lr = dict(base_batch_size=1, enable=False)
backend_args = None
base_lr = 0.004
custom_hooks = [
    dict(
        ema_type='ExpMomentumEMA',
        momentum=0.0002,
        priority=49,
        type='EMAHook',
        update_buffers=True),
]
data_root = ''
dataset_type = 'CocoDataset'
default_hooks = dict(
    checkpoint=dict(
        interval=5,
        max_keep_ckpts=2,
        rule='greater',
        save_best='coco/bbox_mAP',
        type='CheckpointHook'),
    logger=dict(interval=40, type='LoggerHook'),
    param_scheduler=dict(type='ParamSchedulerHook'),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    timer=dict(type='IterTimerHook'),
    visualization=dict(type='DetVisualizationHook'))
default_scope = 'mmdet'
env_cfg = dict(
    cudnn_benchmark=False,
    dist_cfg=dict(backend='nccl'),
    mp_cfg=dict(mp_start_method='spawn', opencv_num_threads=0))
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
iou_threshold = 0.5

log_level = 'INFO'
log_processor = dict(by_epoch=True, type='LogProcessor', window_size=50)
max_epochs = 100
max_per_img = 400
metainfo = dict(
    classes=('Mitochondria', ), palette=[
        (
            46,
            139,
            87,
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
            123.0,
            123.0,
            123.0,
        ],
        std=[
            53.0,
            53.0,
            53.0,
        ],
        type='DetDataPreprocessor'),
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
    test_cfg=dict(
        mask_thr_binary=0.5,
        max_per_img=150,
        min_bbox_size=0,
        nms=dict(iou_threshold=0.6, type='soft_nms'),
        nms_pre=500,
        score_thr=0.05),
    train_cfg=dict(
        allowed_border=-1,
        assigner=dict(topk=13, type='DynamicSoftLabelAssigner'),
        debug=False,
        pos_weight=-1),
    type='RTMDet')
optim_wrapper = dict(
    optimizer=dict(lr=5e-05, type='AdamW', weight_decay=0.05),
    paramwise_cfg=dict(
        bias_decay_mult=0, bypass_duplicate=True, norm_decay_mult=0),
    type='OptimWrapper')
param_scheduler = [
    dict(begin=0, by_epoch=True, end=10, start_factor=0.1, type='LinearLR'),
]
resume = False
stage2_num_epochs = 80
test_cfg = dict(type='TestLoop')
test_dataloader = dict(
    batch_size=1,
    dataset=dict(
        ann_file=
        'val.json',
        data_prefix=dict(img=''),
        data_root=
        '',
        metainfo=dict(classes=('Mitochondria', ), palette=[
            (
                46,
                139,
                87,
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
            dict(
                poly2mask=False,
                type='LoadAnnotations',
                with_bbox=True,
                with_mask=True),
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
        type='CocoDataset'),
    num_workers=0,
    persistent_workers=False)
test_evaluator = dict(score_thr=0.05, type='BinaryInsSegMetric')
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
train_cfg = dict(max_epochs=100, type='EpochBasedTrainLoop', val_interval=5)
train_dataloader = dict(
    batch_size=1,
    dataset=dict(
        dataset=dict(
            ann_file=
            '',
            data_prefix=dict(img=''),
            data_root=
            '',
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
            type='CocoDataset'),
        times=8,
        type='RepeatDataset'),
    num_workers=0,
    persistent_workers=False,
    pin_memory=False)
train_pipeline = [
    dict(backend_args=None, type='LoadImageFromFile'),
    dict(
        poly2mask=False,
        type='LoadAnnotations',
        with_bbox=True,
        with_mask=True),
    dict(img_scale=(
        512,
        512,
    ), pad_val=114.0, type='CachedMosaic'),
    dict(
        keep_ratio=True,
        ratio_range=(
            0.1,
            2.0,
        ),
        scale=(
            1280,
            1280,
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
    dict(type='YOLOXHSVRandomAug'),
    dict(prob=0.5, type='RandomFlip'),
    dict(pad_val=dict(img=(
        114,
        114,
        114,
    )), size=(
        512,
        512,
    ), type='Pad'),
    dict(
        img_scale=(
            512,
            512,
        ),
        max_cached_images=20,
        pad_val=(
            114,
            114,
            114,
        ),
        ratio_range=(
            1.0,
            1.0,
        ),
        type='CachedMixUp'),
    dict(min_gt_bbox_wh=(
        1,
        1,
    ), type='FilterAnnotations'),
    dict(type='PackDetInputs'),
]
train_pipeline_stage2 = [
    dict(backend_args=None, type='LoadImageFromFile'),
    dict(
        poly2mask=False,
        type='LoadAnnotations',
        with_bbox=True,
        with_mask=True),
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
    dict(pad_val=dict(img=(
        114,
        114,
        114,
    )), size=(
        512,
        512,
    ), type='Pad'),
    dict(type='PackDetInputs'),
]
tta_model = dict(
    tta_cfg=dict(
        max_per_img=400, nms=dict(iou_threshold=0.6, type='soft_nms')),
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
                        512,
                        512,
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
    batch_size=1,
    dataset=dict(
        ann_file=
        '',
        data_prefix=dict(img=''),
        data_root=
        '',
        metainfo=dict(classes=('Mitochondria', ), palette=[
            (
                46,
                139,
                87,
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
            dict(
                poly2mask=False,
                type='LoadAnnotations',
                with_bbox=True,
                with_mask=True),
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
        type='CocoDataset'),
    num_workers=0,
    persistent_workers=False)
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
work_dir = '{{fileDirname}}/../../../save_logs/ViT_RTM_mito'

# EMCFsys dataset override. The dataset contains image/ and the three COCO
# annotation files directly under datasets_temp/MitoInstanceSegDataset.
_data_root = '{{fileDirname}}/../../../datasets_temp/LeeMitoInsSeg'
_train_ann = 'train.json'
_val_ann = 'val.json'
data_root = _data_root
load_from = None
resume = False

for _loader_name, _ann_name in (
        ('val_dataloader', _val_ann), ('test_dataloader', 'test.json')):
    _loader = globals()[_loader_name]
    _dataset = _loader['dataset']
    _dataset['type'] = 'CocoDataset'
    _dataset['data_root'] = _data_root
    _dataset['ann_file'] = _ann_name
    _dataset['data_prefix'] = dict(img='image/')
    _dataset['metainfo'] = metainfo
    _loader['num_workers'] = 0
    _loader['persistent_workers'] = False

_train_dataset = train_dataloader['dataset']['dataset']
_train_dataset['type'] = 'CocoDataset'
_train_dataset['data_root'] = _data_root
_train_dataset['ann_file'] = _train_ann
_train_dataset['data_prefix'] = dict(img='image/')
_train_dataset['metainfo'] = metainfo
train_dataloader['num_workers'] = 0
train_dataloader['persistent_workers'] = False
train_dataloader['pin_memory'] = False

# val_evaluator = dict(score_thr=0.05, type='BinaryInsSegMetric')
test_evaluator = dict(score_thr=0.05, type='BinaryInsSegMetric')
