from .transforms import (
    Compose,
    LoadImage,
    LoadMask,
    Normalize,
    PhotometricDistortion,
    RandomCropWithCategoryRatio,
    RandomFlip,
    RandomResizeKeepRatio,
    Resize,
    SegmentationPad,
    ToTensor,
)


MMSEG_MEAN = (123.0, 123.0, 123.0)
MMSEG_STD = (53.0, 53.0, 53.0)


def get_train_transform(
    target_size=(512, 512),
    scale=(512, 512),
    ratio_range=(0.5, 2.0),
    cat_max_ratio=0.75,
    ignore_index=None,
):
    """Build the training pipeline from ``MAE_EMCF_Unet.py``."""

    seg_pad_val = ignore_index if ignore_index is not None else 0
    return Compose(
        [
            LoadImage(),
            LoadMask(),
            RandomResizeKeepRatio(scale=scale, ratio_range=ratio_range),
            RandomCropWithCategoryRatio(
                target_size,
                cat_max_ratio=cat_max_ratio,
                ignore_index=ignore_index,
                seg_pad_val=seg_pad_val,
            ),
            RandomFlip(flip_ratio=0.5, direction="horizontal"),
            SegmentationPad(target_size, pad_val=0, seg_pad_val=seg_pad_val),
            PhotometricDistortion(),
            Normalize(mean=MMSEG_MEAN, std=MMSEG_STD),
            ToTensor(),
        ]
    )


def get_val_transform(target_size=(512, 512)):
    """Build the deterministic validation/test pipeline from the mmseg config."""

    return Compose(
        [
            LoadImage(),
            LoadMask(),
            Resize(target_size),
            Normalize(mean=MMSEG_MEAN, std=MMSEG_STD),
            ToTensor(),
        ]
    )

