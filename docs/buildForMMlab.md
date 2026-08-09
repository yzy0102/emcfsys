## If you want to use mmlab-backend version for EMCellFound finetuning. Please install the mmbackend package first

```Shell
pip install -U openmim

mim install mmengine

pip install mmcv==2.0.0rc4 --no-build-isolation

mim install mmdet
mim install segmentation

pip install ftfy

You can check the backend using in your environment:
python -c "import mmcv, mmengine, mmseg, mmdet; from mmcv.ops import MultiScaleDeformableAttention; print(mmcv.__version__, mmengine.__version__, mmseg.__version__, mmdet.__version__)"
```

Then follow the [mmLab_backend_demo.ipynb](../mmLab_backend_demo.ipynb)
