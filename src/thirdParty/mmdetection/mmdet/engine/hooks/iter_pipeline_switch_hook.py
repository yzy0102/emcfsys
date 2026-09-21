"""Switch a dataset pipeline at a training iteration.

PipelineSwitchHook is epoch-based and cannot switch midway through an
IterBasedTrainLoop, which runs as one long epoch.
"""

from mmcv.transforms import Compose
from mmengine.hooks import Hook

from mmdet.registry import HOOKS


@HOOKS.register_module()
class IterPipelineSwitchHook(Hook):
    """Use a second data pipeline beginning with switch_iter.

    The training dataloader must use num_workers=0. With worker processes,
    changing the dataset in the parent does not change already-prefetched
    batches in those workers.
    """

    def __init__(self, switch_iter: int, switch_pipeline: list[dict]):
        if switch_iter < 0:
            raise ValueError('switch_iter must not be negative')
        self.switch_iter = switch_iter
        self.switch_pipeline = switch_pipeline
        self._has_switched = False

    def _switch(self, runner) -> None:
        if self._has_switched:
            return
        loader = runner.train_dataloader
        if loader.num_workers != 0:
            raise RuntimeError(
                'IterPipelineSwitchHook requires train_dataloader.num_workers=0')
        loader.dataset.pipeline = Compose(self.switch_pipeline)
        self._has_switched = True
        runner.logger.info(
            'Switched training pipeline at iteration %d', self.switch_iter)

    def before_train(self, runner) -> None:
        # Checkpoint resume can start on or after the switch iteration.
        if runner.iter >= self.switch_iter:
            self._switch(runner)

    def after_train_iter(self, runner, batch_idx: int, data_batch, outputs=None) -> None:
        # IterBasedTrainLoop increments runner.iter after this hook. Switching
        # here makes the very next fetched batch use the second pipeline.
        if runner.iter + 1 >= self.switch_iter:
            self._switch(runner)
