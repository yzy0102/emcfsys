_base_ = [
    '../_base_/default_runtime.py', 
    '../_base_/datasets/coco_detection.py'
]
custom_imports = dict(
    imports=[
        'mmdet.models.detectors.sliding_rtmdet',
        'mmdet.engine.hooks.iter_pipeline_switch_hook',
    ],
    allow_failed_imports=False)
model = dict(
    type='SlidingRTMDet',
    data_preprocessor=dict(
        type='DetDataPreprocessor', pad_size_divisor=32, pad_value=114),
    slide_cfg=dict(
        crop_size=(512, 512),
        stride=(400, 400),
        merge_nms=dict(type='nms', iou_threshold=0.5),
        max_per_img=500),
    backbone=dict(
        img_size = 512,
        patch_size=16, 
        in_chans=3, 
        embed_dim=768,
        cffn_ratio=0.25,
        conv_inplane=64,
        deform_num_heads=6,
        deform_ratio=1.0,
        depth=12,
        drop_path_rate=0.1,
        # embed_dim=192,
        interaction_indexes=[
            [
                0,
                2,
            ],
            [
                3,
                5,
            ],
            [
                6,
                8,
            ],
            [
                9,
                11,
            ],
        ],
        layer_scale=False,
        mlp_ratio=4,
        n_points=4,
        num_heads=3,
        out_indices=[
            1,
            2,
            3,
        ],
        type='ViTAdapter',
        window_attn=[
            True,
            True,
            False,
            True,
            True,
            False,
            True,
            True,
            False,
            True,
            True,
            False,
        ],
        window_size=[
            14,
            14,
            None,
            14,
            14,
            None,
            14,
            14,
            None,
            14,
            14,
            None,
        ]),
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
    train_cfg=dict(
        assigner=dict(type='DynamicSoftLabelAssigner', topk=13),
        allowed_border=-1,
        pos_weight=-1,
        debug=False),
    test_cfg=dict(
        nms_pre=30000,
        min_bbox_size=0,
        score_thr=0.001,
        mask_thr_binary=0.5,
        nms=dict(type='nms', iou_threshold=0.65),
        max_per_img=500),
)

train_pipeline = [
    dict(type='LoadImageFromFile', backend_args={{_base_.backend_args}}),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
    dict(type='RandomCrop', crop_size=(640, 640)),
    dict(
        type='RandomResize',
        scale=(640, 640),
        ratio_range=(.8, 1.5),
        keep_ratio=True),
    dict(type='RandomCrop', crop_size=(512, 512)),
    dict(type='YOLOXHSVRandomAug'),
    dict(type='RandomFlip', prob=0.5),
    dict(type='Pad', size=(512, 512), pad_val=dict(img=(114, 114, 114))),
    dict(type='PackDetInputs')
]

train_pipeline_stage2 = [
    dict(type='LoadImageFromFile', backend_args={{_base_.backend_args}}),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
    dict(type='RandomCrop', crop_size=(512, 512)),
    dict(
        type='RandomResize',
        scale=(512, 512),
        ratio_range=(0.1, 2.0),
        keep_ratio=True),
    dict(type='RandomCrop', crop_size=(512, 512)),
    dict(type='YOLOXHSVRandomAug'),
    dict(type='RandomFlip', prob=0.5),
    dict(type='Pad', size=(512, 512), pad_val=dict(img=(114, 114, 114))),
    dict(type='PackDetInputs')
]

test_pipeline = [
    dict(type='LoadImageFromFile', backend_args={{_base_.backend_args}}),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                    'scale_factor'))
]

train_dataloader = dict(
    batch_size=1,
    # Changing the pipeline inside worker processes would leave prefetched
    # batches on the old pipeline after the switch iteration.
    num_workers=0,
    sampler=dict(type='InfiniteSampler', shuffle=True),
    batch_sampler=None,
    pin_memory=True,
    dataset=dict(pipeline=train_pipeline))
val_dataloader = dict(
    batch_size=1, num_workers=2, dataset=dict(pipeline=test_pipeline))
test_dataloader = dict(
    batch_size=1, num_workers=2, dataset=dict(pipeline=test_pipeline))

# Iteration count controls how many times random crops are sampled from the
# training images; adjust max_iters to fit the available training budget.
max_iters = 3000
stage2_num_iters = 200
base_lr = 0.0001
val_interval = 300
checkpoint_interval = 300
warmup_iters = 300

train_cfg = dict(
    type='IterBasedTrainLoop',
    max_iters=max_iters,
    val_interval=val_interval)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')
val_evaluator = dict(
    metric=['bbox', 'segm'],
    proposal_nums=(400, 700, 1000),
    use_dense_max_dets=True)
test_evaluator = dict(
    metric=['bbox', 'segm'],
    proposal_nums=(400, 700, 1000),
    use_dense_max_dets=True)

norm_cfg = dict(requires_grad=True, type='BN')
optim_wrapper = dict(
    clip_grad=dict(max_norm=1.0),
    optimizer=dict(lr=base_lr, type='AdamW', weight_decay=0.01),
    paramwise_cfg=dict(
        bias_decay_mult=0.0,
        bypass_duplicate=True,
        custom_keys=dict(
            auxiliary_head=dict(lr_mult=10.0),
            backbone=dict(lr_mult=0.1),
            neck = dict(lr_mult=10.0),
            bbox_head=dict(lr_mult=10.0)),
        norm_decay_mult=0.0),
    type='OptimWrapper')

auto_scale_lr = dict(enable=False, base_batch_size=2)

# # optimizer
# optim_wrapper = dict(

#     type='OptimWrapper',
#     optimizer=dict(type='AdamW', lr=base_lr, weight_decay=0.05),
#     paramwise_cfg=dict(
#         norm_decay_mult=0, bias_decay_mult=0, bypass_duplicate=True))

# learning rate
param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=1.0e-5,
        by_epoch=False,
        begin=0,
        end=warmup_iters),
    dict(
        # Retain the original constant-LR first half, then cosine decay.
        type='CosineAnnealingLR',
        eta_min=base_lr * 0.05,
        begin=max_iters // 2,
        end=max_iters,
        T_max=max_iters // 2,
        by_epoch=False),
]

# hooks


custom_hooks = [
    dict(
        type='EMAHook',
        ema_type='ExpMomentumEMA',
        momentum=0.0002,
        update_buffers=True,
        priority=49),
    dict(
        type='IterPipelineSwitchHook',
        switch_iter=max_iters - stage2_num_iters,
        switch_pipeline=train_pipeline_stage2)
]

log_processor = dict(by_epoch=False)

default_hooks = dict(
    checkpoint=dict(
        type="CheckpointHook",
        by_epoch=False,
        interval=checkpoint_interval,
        max_keep_ckpts=1,
        save_best="coco/bbox_mAP_75",
    ),
    logger=dict(
        type="LoggerHook",
        interval=20,
        log_metric_by_epoch=False,
    ),
    param_scheduler=dict(type="ParamSchedulerHook"),
    sampler_seed=dict(type="DistSamplerSeedHook"),
    timer=dict(type="IterTimerHook"),

)


# The original-size COCO images remain one evaluation sample each. The model
# creates windows internally and returns merged full-image boxes and masks.
data_root = '{{fileDirname}}/../../../../../datasets_temp/MitoInstanceSegDataset'
for split, loader in (('train', train_dataloader),
                      ('val', val_dataloader),
                      ('test', test_dataloader)):
    loader['dataset'].update(
        type='CocoDataset',
        data_root=data_root,
        ann_file=f'{split}.json',
        data_prefix=dict(img='image/'),
        metainfo=dict(classes=('Mitochondria',), palette=[(220, 20, 60)]))
    loader['persistent_workers'] = False
val_evaluator['ann_file'] = f'{data_root}/val.json'
test_evaluator['ann_file'] = f'{data_root}/test.json'
