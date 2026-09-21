from pathlib import Path
from types import SimpleNamespace

import torch
from mmengine import Config
from mmengine.structures import InstanceData

from mmdet.models.detectors.rtmdet import RTMDet
from mmdet.models.detectors.sliding_rtmdet import SlidingRTMDet, window_starts
from mmdet.structures import DetDataSample


def test_sliding_config_uses_full_image_samples_and_mask_metric():
    path = (
        Path(__file__).resolve().parents[1]
        / 'src/thirdParty/mmdetection/configs/rtmdet/RTM_Ins.py'
    )
    cfg = Config.fromfile(str(path))

    assert cfg.model.type == 'SlidingRTMDet'
    assert cfg.model.test_cfg.mask_thr_binary == 0.5
    assert cfg.test_evaluator.metric == ['bbox', 'segm']
    assert cfg.test_dataloader.batch_size == 1
    assert [step['type'] for step in cfg.test_pipeline] == [
        'LoadImageFromFile', 'PackDetInputs'
    ]
    assert Path(cfg.data_root).resolve() == (
        Path(__file__).resolve().parents[1]
        / 'datasets_temp/MitoInstanceSegDataset'
    )
    assert cfg.val_dataloader.dataset.ann_file == 'val.json'
    assert cfg.test_dataloader.dataset.ann_file == 'test.json'
    assert cfg.train_cfg.type == 'IterBasedTrainLoop'
    assert cfg.train_cfg.max_iters == cfg.max_iters
    assert cfg.train_dataloader.sampler.type == 'InfiniteSampler'
    assert cfg.train_dataloader.num_workers == 0
    assert all(not scheduler.by_epoch for scheduler in cfg.param_scheduler)
    assert cfg.default_hooks.checkpoint.by_epoch is False
    assert cfg.log_processor.by_epoch is False
    assert cfg.custom_hooks[-1].type == 'IterPipelineSwitchHook'
    assert cfg.custom_hooks[-1].switch_iter == cfg.max_iters - cfg.stage2_num_iters
    assert cfg.optim_wrapper.optimizer.type == 'AdamW'
    assert 'momentum' not in cfg.optim_wrapper.optimizer
    assert cfg.optim_wrapper.paramwise_cfg.bypass_duplicate is True
    assert cfg.optim_wrapper.optimizer.lr == cfg.base_lr
    assert cfg.param_scheduler[1].eta_min < cfg.base_lr


def test_window_starts_cover_last_edge():
    assert window_starts(1024, 512, 384) == [0, 384, 512]
    assert window_starts(300, 512, 384) == [0]


def test_sliding_prediction_maps_boxes_and_masks_and_merges_duplicates(monkeypatch):
    model = SlidingRTMDet.__new__(SlidingRTMDet)
    torch.nn.Module.__init__(model)
    model.slide_cfg = dict(
        crop_size=(512, 512),
        stride=(384, 384),
        merge_nms=dict(type='nms', iou_threshold=0.5),
        max_per_img=100,
    )
    model.data_preprocessor = SimpleNamespace(pad_value=114)
    tile_calls = []
    objects = ((500, 500), (900, 900))

    def fake_predict(self, batch_inputs, batch_data_samples, rescale=True):
        assert batch_inputs.shape == (1, 3, 512, 512)
        sample = batch_data_samples[0]
        top, left = sample.metainfo['tile_offset']
        tile_calls.append((top, left))
        boxes, masks = [], []
        for x, y in objects:
            x1, y1 = max(x - 20 - left, 0), max(y - 20 - top, 0)
            x2, y2 = min(x + 20 - left, 512), min(y + 20 - top, 512)
            if x2 <= x1 or y2 <= y1:
                continue
            boxes.append([x1, y1, x2, y2])
            mask = torch.zeros((512, 512), dtype=torch.bool)
            mask[y1:y2, x1:x2] = True
            masks.append(mask)
        sample.pred_instances = InstanceData(
            bboxes=torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4),
            scores=torch.full((len(boxes),), 0.9),
            labels=torch.zeros((len(boxes),), dtype=torch.long),
            masks=torch.stack(masks) if masks else torch.zeros((0, 512, 512), dtype=torch.bool),
        )
        return batch_data_samples

    monkeypatch.setattr(RTMDet, 'predict', fake_predict)
    sample = DetDataSample()
    sample.set_metainfo(dict(img_id=7, ori_shape=(1024, 1024)))
    result = model.predict(torch.zeros((1, 3, 1024, 1024)), [sample])
    instances = result[0].pred_instances

    assert len(tile_calls) == 9
    assert result[0] is sample
    assert result[0].metainfo['img_id'] == 7
    assert len(instances) == 2
    assert instances.masks.shape == (2, 1024, 1024)
    assert instances.masks[:, 500, 500].any()
    assert instances.masks[:, 900, 900].any()
    assert torch.allclose(
        instances.bboxes.sort(dim=0).values,
        torch.tensor([[480, 480, 520, 520], [880, 880, 920, 920]], dtype=torch.float32).sort(dim=0).values,
    )


def test_small_image_is_padded_for_the_model_but_keeps_original_mask_shape(monkeypatch):
    model = SlidingRTMDet.__new__(SlidingRTMDet)
    torch.nn.Module.__init__(model)
    model.slide_cfg = dict(crop_size=(512, 512), stride=(384, 384))
    model.data_preprocessor = SimpleNamespace(pad_value=114)
    calls = []

    def fake_predict(self, batch_inputs, batch_data_samples, rescale=True):
        calls.append(batch_inputs)
        sample = batch_data_samples[0]
        assert sample.metainfo['ori_shape'] == (300, 320)
        sample.pred_instances = InstanceData(
            bboxes=torch.empty((0, 4)),
            scores=torch.empty((0,)),
            labels=torch.empty((0,), dtype=torch.long),
            masks=torch.zeros((0, 512, 512), dtype=torch.bool),
        )
        return batch_data_samples

    monkeypatch.setattr(RTMDet, 'predict', fake_predict)
    sample = DetDataSample()
    sample.set_metainfo(dict(ori_shape=(300, 320)))
    instances = model.predict(torch.zeros((1, 3, 300, 320)), [sample])[0].pred_instances

    assert len(calls) == 1
    assert calls[0].shape == (1, 3, 512, 512)
    assert calls[0][0, 0, 400, 400] == 114
    assert instances.masks.shape == (0, 300, 320)


def test_edge_tile_boxes_are_clipped_to_the_original_image(monkeypatch):
    model = SlidingRTMDet.__new__(SlidingRTMDet)
    torch.nn.Module.__init__(model)
    model.slide_cfg = dict(
        crop_size=(512, 512),
        stride=(384, 384),
        merge_nms=dict(type='nms', iou_threshold=0.5),
        max_per_img=10,
    )
    model.data_preprocessor = SimpleNamespace(pad_value=114)

    def fake_predict(self, batch_inputs, batch_data_samples, rescale=True):
        sample = batch_data_samples[0]
        mask = torch.zeros((1, 512, 512), dtype=torch.bool)
        mask[:, 250:350, 250:350] = True
        sample.pred_instances = InstanceData(
            bboxes=torch.tensor([[250.0, 250.0, 350.0, 330.0]]),
            scores=torch.tensor([0.9]),
            labels=torch.tensor([0]),
            masks=mask,
        )
        return batch_data_samples

    monkeypatch.setattr(RTMDet, 'predict', fake_predict)
    sample = DetDataSample()
    sample.set_metainfo(dict(ori_shape=(300, 320)))
    instances = model.predict(torch.zeros((1, 3, 300, 320)), [sample])[0].pred_instances

    assert torch.equal(
        instances.bboxes, torch.tensor([[250.0, 250.0, 320.0, 300.0]])
    )
