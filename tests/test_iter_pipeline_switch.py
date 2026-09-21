import logging
from types import SimpleNamespace

import pytest
from mmcv.transforms import Compose

from mmdet.engine.hooks.iter_pipeline_switch_hook import IterPipelineSwitchHook


def make_runner(iteration=0, num_workers=0):
    original_pipeline = object()
    dataset = SimpleNamespace(pipeline=original_pipeline)
    loader = SimpleNamespace(dataset=dataset, num_workers=num_workers)
    runner = SimpleNamespace(
        iter=iteration, train_dataloader=loader, logger=logging.getLogger(__name__)
    )
    return runner, original_pipeline


def test_switch_after_last_iteration_of_first_stage():
    hook = IterPipelineSwitchHook(switch_iter=5, switch_pipeline=[])
    runner, original_pipeline = make_runner(iteration=3)

    hook.before_train(runner)
    hook.after_train_iter(runner, batch_idx=3, data_batch={})
    assert runner.train_dataloader.dataset.pipeline is original_pipeline

    runner.iter = 4
    hook.after_train_iter(runner, batch_idx=4, data_batch={})
    switched_pipeline = runner.train_dataloader.dataset.pipeline
    assert isinstance(switched_pipeline, Compose)

    runner.iter = 5
    hook.after_train_iter(runner, batch_idx=5, data_batch={})
    assert runner.train_dataloader.dataset.pipeline is switched_pipeline


def test_resume_after_switch_iteration_uses_second_pipeline():
    hook = IterPipelineSwitchHook(switch_iter=5, switch_pipeline=[])
    runner, original_pipeline = make_runner(iteration=7)

    hook.before_train(runner)
    assert isinstance(runner.train_dataloader.dataset.pipeline, Compose)
    assert runner.train_dataloader.dataset.pipeline is not original_pipeline


def test_worker_prefetch_is_rejected_instead_of_using_stale_pipeline():
    hook = IterPipelineSwitchHook(switch_iter=5, switch_pipeline=[])
    runner, original_pipeline = make_runner(iteration=4, num_workers=2)

    with pytest.raises(RuntimeError, match='num_workers=0'):
        hook.after_train_iter(runner, batch_idx=4, data_batch={})
    assert runner.train_dataloader.dataset.pipeline is original_pipeline
