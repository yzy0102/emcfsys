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

# learning policy
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
# training schedule for 100 epoch
train_cfg = dict(max_epochs=100, type='EpochBasedTrainLoop', val_interval=2)

val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

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
