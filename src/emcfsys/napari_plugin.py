from napari_plugin_engine import napari_hook_implementation
from .widgets.single_infer import SingleInferWidget
from .widgets.batch_infer import BatchInferWidget
from .widgets.train import TrainWidget

@napari_hook_implementation
def napari_experimental_provide_dock_widget():
    return [
        (SingleInferWidget, {"name": "Single Image Inference"}),
        (BatchInferWidget, {"name": "Batch Inference"}),
        (TrainWidget, {"name": "Model Training"}),
    ]
