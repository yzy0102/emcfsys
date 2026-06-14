import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
from PIL import Image

from emcfsys.EMCellFound.datasets import COCOInstanceSegmentationDataset
from emcfsys.EMCellFound.models.RTMInstanceSeg import (
    EMCellFoundMask2FormerInstanceSegmenter,
    EMCellFoundMaskRCNNInstanceSegmenter,
    EMCellFoundRTMInstanceSegmenter,
    MODEL_CONDINST_INSTANCE,
    MODEL_MASK_RCNN_INSTANCE,
    MODEL_MASK2FORMER_INSTANCE,
    MODEL_RTM_INSTANCE_LARGE,
    MODEL_RTM_INSTANCE_TINY,
    MODEL_SOLOV2_INSTANCE,
    MODEL_YOLACT_INSTANCE,
)
from emcfsys.EMCellFound.models.RTMInstanceSeg import batched_nms
from emcfsys.EMCellFound.models.model_factory import get_model
from emcfsys.utils import instance_segmentation_tasks as ist
from emcfsys.utils.instance_segmentation_tasks import (
    INSTANCE_MODEL_CHOICES,
    InstanceSegmentationInferenceRequest,
    InstanceSegmentationTrainingRequest,
    iter_instance_segmentation_training_task,
    prediction_to_instance_mask,
    run_instance_segmentation_inference_task,
    run_instance_segmentation_training_task,
)
from emcfsys.utils.coco_instance_inspector import (
    check_coco_instance_dataset,
    format_coco_instance_check_report,
    load_coco_instance_preview,
)
from emcfsys.utils.labelme_coco_tasks import (
    LabelMeInstanceToCOCORequest,
    convert_labelme_instance_folder_to_coco,
)


def _write_tiny_coco_dataset(root):
    image_dir = root / "images"
    image_dir.mkdir(parents=True)
    Image.fromarray(np.zeros((16, 16, 3), dtype=np.uint8)).save(image_dir / "img.png")
    annotations = {
        "images": [
            {"id": 1, "file_name": "img.png", "width": 16, "height": 16},
        ],
        "categories": [
            {"id": 3, "name": "mitochondria"},
        ],
        "annotations": [
            {
                "id": 10,
                "image_id": 1,
                "category_id": 3,
                "bbox": [2, 3, 8, 6],
                "area": 48,
                "iscrowd": 0,
                "segmentation": [[2, 3, 10, 3, 10, 9, 2, 9]],
            }
        ],
    }
    annotation_path = root / "instances.json"
    annotation_path.write_text(json.dumps(annotations), encoding="utf-8")
    return image_dir, annotation_path


def _write_tiny_labelme_dataset(root):
    root.mkdir(parents=True)
    image_path = root / "img.png"
    Image.fromarray(np.zeros((16, 16, 3), dtype=np.uint8)).save(image_path)
    labelme = {
        "version": "5.0.0",
        "imagePath": image_path.name,
        "imageHeight": 16,
        "imageWidth": 16,
        "shapes": [
            {
                "label": "Mito",
                "points": [[2, 3], [10, 3], [10, 9], [2, 9]],
                "shape_type": "polygon",
            }
        ],
    }
    (root / "img.json").write_text(json.dumps(labelme), encoding="utf-8")
    return root


def _write_tiny_labelme_rectangle_dataset(root):
    root.mkdir(parents=True)
    image_path = root / "rect.png"
    Image.fromarray(np.zeros((16, 16, 3), dtype=np.uint8)).save(image_path)
    labelme = {
        "version": "5.0.0",
        "imagePath": image_path.name,
        "imageHeight": 16,
        "imageWidth": 16,
        "shapes": [
            {
                "label": "Mito",
                "points": [[2, 3], [10, 9]],
                "shape_type": "rectangle",
            }
        ],
    }
    (root / "rect.json").write_text(json.dumps(labelme), encoding="utf-8")
    return root


def _write_labelme_split_dataset(root, num_images=5):
    root.mkdir(parents=True)
    for index in range(num_images):
        image_name = f"img_{index}.png"
        Image.fromarray(np.zeros((16, 16, 3), dtype=np.uint8)).save(root / image_name)
        labelme = {
            "version": "5.0.0",
            "imagePath": image_name,
            "imageHeight": 16,
            "imageWidth": 16,
            "shapes": [
                {
                    "label": "Mito",
                    "points": [[2, 3], [10, 3], [10, 9], [2, 9]],
                    "shape_type": "polygon",
                }
            ],
        }
        (root / f"img_{index}.json").write_text(
            json.dumps(labelme),
            encoding="utf-8",
        )
    return root


def test_coco_instance_dataset_reads_polygon_annotations(tmp_path):
    image_dir, annotation_path = _write_tiny_coco_dataset(tmp_path)

    dataset = COCOInstanceSegmentationDataset(
        image_dir,
        annotation_path,
        img_size=32,
    )
    image, target = dataset[0]

    assert dataset.class_names == ["mitochondria"]
    assert image.shape == (3, 32, 32)
    assert target["boxes"].shape == (1, 4)
    assert target["labels"].tolist() == [1]
    assert target["masks"].shape == (1, 32, 32)
    assert int(target["masks"].sum()) > 0


def test_coco_instance_inspector_reports_and_previews_dataset(tmp_path):
    image_dir, annotation_path = _write_tiny_coco_dataset(tmp_path)

    report = check_coco_instance_dataset(image_dir, annotation_path)
    text_report = format_coco_instance_check_report(report)
    preview = load_coco_instance_preview(image_dir, annotation_path, index=0)

    assert report["ok"] is True
    assert report["summary"]["num_images"] == 1
    assert report["summary"]["num_existing_images"] == 1
    assert report["summary"]["num_trainable_annotations"] == 1
    assert report["categories"] == [
        {"id": 3, "name": "mitochondria", "annotation_count": 1}
    ]
    assert "COCO instance dataset check: OK" in text_report
    assert preview["image"].shape == (16, 16, 3)
    assert preview["instance_mask"].shape == (16, 16)
    assert int(preview["instance_mask"].max()) == 1
    assert preview["labels"] == ["mitochondria"]
    assert preview["boxes"].shape == (1, 4)


def test_labelme_instance_folder_converts_to_coco_and_dataset(tmp_path):
    labelme_dir = _write_tiny_labelme_dataset(tmp_path / "labelme")
    output_json = tmp_path / "coco" / "train.json"
    image_output_dir = tmp_path / "coco" / "images"

    result = convert_labelme_instance_folder_to_coco(
        LabelMeInstanceToCOCORequest(
            labelme_json_dir=str(labelme_dir),
            output_json=str(output_json),
            image_output_dir=str(image_output_dir),
            copy_images=True,
            category_names=["Mito"],
        )
    )

    assert result["num_images"] == 1
    assert result["num_annotations"] == 1
    assert result["categories"] == ["Mito"]
    assert (image_output_dir / "img.png").exists()

    coco = json.loads(output_json.read_text(encoding="utf-8"))
    assert coco["images"][0]["file_name"] == "img.png"
    assert coco["annotations"][0]["bbox"] == [2.0, 3.0, 8.0, 6.0]
    assert coco["annotations"][0]["segmentation"] == [[2.0, 3.0, 10.0, 3.0, 10.0, 9.0, 2.0, 9.0]]

    dataset = COCOInstanceSegmentationDataset(image_output_dir, output_json, img_size=16)
    _, target = dataset[0]
    assert target["labels"].tolist() == [1]
    assert target["masks"].shape == (1, 16, 16)


def test_labelme_instance_converter_defaults_image_folder_and_rectangle(tmp_path):
    labelme_dir = _write_tiny_labelme_rectangle_dataset(tmp_path / "labelme")
    output_json = tmp_path / "coco" / "train.json"

    result = convert_labelme_instance_folder_to_coco(
        LabelMeInstanceToCOCORequest(
            labelme_json_dir=str(labelme_dir),
            output_json=str(output_json),
            copy_images=True,
        )
    )

    image_output_dir = tmp_path / "coco" / "images"
    coco = json.loads(output_json.read_text(encoding="utf-8"))

    assert result["image_output_dir"] == str(image_output_dir)
    assert result["categories"] == ["Mito"]
    assert (image_output_dir / "rect.png").exists()
    assert coco["annotations"][0]["bbox"] == [2.0, 3.0, 8.0, 6.0]
    assert coco["annotations"][0]["segmentation"] == [
        [2.0, 3.0, 10.0, 3.0, 10.0, 9.0, 2.0, 9.0]
    ]


def test_labelme_instance_converter_writes_train_val_test_splits(tmp_path):
    labelme_dir = _write_labelme_split_dataset(tmp_path / "labelme", num_images=5)
    output_json = tmp_path / "coco" / "instances.json"
    image_output_dir = tmp_path / "coco" / "images"

    result = convert_labelme_instance_folder_to_coco(
        LabelMeInstanceToCOCORequest(
            labelme_json_dir=str(labelme_dir),
            output_json=str(output_json),
            image_output_dir=str(image_output_dir),
            copy_images=True,
            split_dataset=True,
            train_ratio=0.6,
            val_ratio=0.2,
            test_ratio=0.2,
            split_seed=7,
        )
    )

    assert result["output_json"] == str(tmp_path / "coco" / "train.json")
    assert result["output_jsons"] == {
        "train": str(tmp_path / "coco" / "train.json"),
        "val": str(tmp_path / "coco" / "val.json"),
        "test": str(tmp_path / "coco" / "test.json"),
    }
    assert result["split_counts"] == {
        "train": {"images": 3, "annotations": 3},
        "val": {"images": 1, "annotations": 1},
        "test": {"images": 1, "annotations": 1},
    }

    seen_image_ids = set()
    for split_name, json_path in result["output_jsons"].items():
        coco = json.loads(Path(json_path).read_text(encoding="utf-8"))
        image_ids = {image["id"] for image in coco["images"]}
        annotation_image_ids = {
            annotation["image_id"] for annotation in coco["annotations"]
        }

        assert image_ids.isdisjoint(seen_image_ids)
        assert annotation_image_ids <= image_ids
        assert coco["categories"] == [{"id": 1, "name": "Mito"}]
        assert len(coco["images"]) == result["split_counts"][split_name]["images"]
        assert len(coco["annotations"]) == result["split_counts"][split_name]["annotations"]
        seen_image_ids.update(image_ids)

    assert seen_image_ids == {1, 2, 3, 4, 5}
    assert len(list(image_output_dir.glob("*.png"))) == 5

    dataset = COCOInstanceSegmentationDataset(
        image_output_dir,
        result["output_jsons"]["train"],
        img_size=16,
    )
    assert len(dataset) == 3


def test_rtm_instance_segmenter_forward_shapes():
    model = EMCellFoundRTMInstanceSegmenter(
        backbone_name="resnet34",
        num_classes=2,
        img_size=64,
        pretrained=False,
        neck_channels=16,
        head_channels=16,
        num_prototypes=8,
    )
    model.eval()

    with torch.no_grad():
        outputs = model(torch.zeros(1, 3, 64, 64))

    assert set(outputs) == {
        "cls_logits",
        "objectness",
        "box_regression",
        "mask_coefficients",
        "mask_prototypes",
    }
    assert len(outputs["cls_logits"]) == 4
    assert outputs["cls_logits"][0].shape[1] == 2
    assert outputs["objectness"][0].shape[1] == 1
    assert outputs["box_regression"][0].shape[1] == 4
    assert outputs["mask_coefficients"][0].shape[1] == 8
    assert outputs["mask_prototypes"].shape[1] == 8


def test_rtm_instance_segmenter_predict_outputs_instances():
    model = EMCellFoundRTMInstanceSegmenter(
        backbone_name="resnet34",
        num_classes=2,
        img_size=64,
        pretrained=False,
        neck_channels=16,
        head_channels=16,
        num_prototypes=8,
    )
    model.eval()

    with torch.no_grad():
        results = model.predict(
            torch.zeros(1, 3, 64, 64),
            score_threshold=0.0,
            max_detections=5,
        )

    assert len(results) == 1
    assert results[0]["boxes"].shape == (5, 4)
    assert results[0]["scores"].shape == (5,)
    assert results[0]["labels"].shape == (5,)
    assert results[0]["masks"].shape == (5, 64, 64)
    assert results[0]["masks"].dtype == torch.bool


def test_rtm_instance_segmenter_loss_backpropagates():
    model = EMCellFoundRTMInstanceSegmenter(
        backbone_name="resnet34",
        num_classes=1,
        img_size=64,
        pretrained=False,
        neck_channels=16,
        head_channels=16,
        num_prototypes=8,
    )
    model.train()
    image = torch.zeros(1, 3, 64, 64)
    target = {
        "boxes": torch.tensor([[12.0, 12.0, 40.0, 42.0]]),
        "labels": torch.tensor([1]),
        "masks": torch.zeros(1, 64, 64, dtype=torch.uint8),
    }
    target["masks"][0, 12:42, 12:40] = 1

    losses = model.loss(image, [target])
    losses["loss"].backward()

    assert torch.isfinite(losses["loss"])
    assert losses["num_pos"].item() >= 1
    assert any(
        parameter.grad is not None
        for parameter in model.parameters()
        if parameter.requires_grad
    )


def test_rtm_instance_segmenter_uses_advanced_mask_losses():
    model = EMCellFoundRTMInstanceSegmenter(
        backbone_name="resnet34",
        num_classes=1,
        img_size=64,
        pretrained=False,
        neck_channels=16,
        head_channels=16,
        num_prototypes=8,
        boundary_loss_weight=0.2,
        focal_mask_loss_weight=0.3,
        tversky_loss_weight=0.4,
    )
    image = torch.zeros(1, 3, 64, 64)
    target = {
        "boxes": torch.tensor([[12.0, 12.0, 40.0, 42.0]]),
        "labels": torch.tensor([1]),
        "masks": torch.zeros(1, 64, 64, dtype=torch.uint8),
    }
    target["masks"][0, 12:42, 12:40] = 1

    losses = model.loss(image, [target])
    losses["loss"].backward()

    assert model.boundary_loss_weight == 0.2
    assert model.focal_mask_loss_weight == 0.3
    assert model.tversky_loss_weight == 0.4
    assert torch.isfinite(losses["loss"])


def test_batched_nms_filters_same_class_overlaps():
    boxes = torch.tensor(
        [
            [0.0, 0.0, 10.0, 10.0],
            [1.0, 1.0, 11.0, 11.0],
            [1.0, 1.0, 11.0, 11.0],
        ]
    )
    scores = torch.tensor([0.9, 0.8, 0.7])
    labels = torch.tensor([1, 1, 2])

    keep = batched_nms(boxes, scores, labels, iou_threshold=0.5)

    assert keep.tolist() == [0, 2]


def test_model_factory_creates_rtm_instance_model():
    model = get_model(
        "rtm_instance",
        backbone_name="resnet34",
        img_size=64,
        num_classes=2,
        pretrained=False,
        neck_channels=16,
        head_channels=16,
        num_prototypes=8,
    )

    assert isinstance(model, EMCellFoundRTMInstanceSegmenter)


def test_model_factory_creates_rtm_family_variants():
    tiny = get_model(
        MODEL_RTM_INSTANCE_TINY,
        backbone_name="resnet34",
        img_size=64,
        num_classes=2,
        pretrained=False,
    )
    large = get_model(
        MODEL_RTM_INSTANCE_LARGE,
        backbone_name="resnet34",
        img_size=64,
        num_classes=2,
        pretrained=False,
        neck_channels=32,
        head_channels=32,
        num_prototypes=8,
    )

    assert isinstance(tiny, EMCellFoundRTMInstanceSegmenter)
    assert isinstance(large, EMCellFoundRTMInstanceSegmenter)
    assert tiny.neck_channels == 64
    assert large.num_prototypes == 8


def test_instance_segmentation_task_builder_supports_new_heads():
    expected_names = {
        "rtm_instance",
        MODEL_RTM_INSTANCE_TINY,
        MODEL_RTM_INSTANCE_LARGE,
        MODEL_YOLACT_INSTANCE,
        MODEL_MASK_RCNN_INSTANCE,
        MODEL_CONDINST_INSTANCE,
        MODEL_SOLOV2_INSTANCE,
        MODEL_MASK2FORMER_INSTANCE,
    }

    assert expected_names.issubset(set(INSTANCE_MODEL_CHOICES))
    yolact_model = ist._build_instance_model(
        model_name=MODEL_YOLACT_INSTANCE,
        backbone_name="resnet34",
        num_classes=1,
        img_size=64,
        pretrained=False,
        model_kwargs={
            "neck_channels": 16,
            "head_channels": 16,
            "num_prototypes": 8,
        },
    )
    mask_rcnn_model = ist._build_instance_model(
        model_name=MODEL_MASK_RCNN_INSTANCE,
        backbone_name="resnet34",
        num_classes=1,
        img_size=64,
        pretrained=False,
        model_kwargs={
            "neck_channels": 16,
            "head_channels": 16,
            "proposal_topk": 8,
            "mask_size": 14,
        },
    )

    assert yolact_model.head_type == "yolact"
    assert isinstance(mask_rcnn_model, EMCellFoundMaskRCNNInstanceSegmenter)


def test_condinst_and_solov2_instance_heads_use_dense_contract():
    for model_name, expected_head in [
        (MODEL_CONDINST_INSTANCE, "condinst"),
        (MODEL_SOLOV2_INSTANCE, "solov2"),
    ]:
        model = get_model(
            model_name,
            backbone_name="resnet34",
            img_size=64,
            num_classes=2,
            pretrained=False,
            neck_channels=16,
            head_channels=16,
            num_prototypes=8,
        )
        model.eval()

        with torch.no_grad():
            outputs = model(torch.zeros(1, 3, 64, 64))
            results = model.predict(
                torch.zeros(1, 3, 64, 64),
                score_threshold=0.0,
                max_detections=3,
            )

        assert isinstance(model, EMCellFoundRTMInstanceSegmenter)
        assert model.head_type == expected_head
        assert outputs["mask_prototypes"].shape[1] == 8
        assert results[0]["boxes"].shape == (3, 4)
        assert results[0]["masks"].shape == (3, 64, 64)


def test_yolact_instance_segmenter_uses_dense_prediction_contract():
    model = get_model(
        MODEL_YOLACT_INSTANCE,
        backbone_name="resnet34",
        img_size=64,
        num_classes=2,
        pretrained=False,
        neck_channels=16,
        head_channels=16,
        num_prototypes=8,
    )
    model.eval()

    with torch.no_grad():
        outputs = model(torch.zeros(1, 3, 64, 64))
        results = model.predict(
            torch.zeros(1, 3, 64, 64),
            score_threshold=0.0,
            max_detections=3,
        )

    assert isinstance(model, EMCellFoundRTMInstanceSegmenter)
    assert model.head_type == "yolact"
    assert outputs["mask_prototypes"].shape[1] == 8
    assert results[0]["boxes"].shape == (3, 4)
    assert results[0]["masks"].shape == (3, 64, 64)


def test_mask_rcnn_instance_segmenter_predict_and_loss_contract():
    model = get_model(
        MODEL_MASK_RCNN_INSTANCE,
        backbone_name="resnet34",
        img_size=64,
        num_classes=1,
        pretrained=False,
        neck_channels=16,
        head_channels=16,
        proposal_topk=8,
        mask_size=14,
    )
    image = torch.zeros(1, 3, 64, 64)
    target = {
        "boxes": torch.tensor([[12.0, 12.0, 40.0, 42.0]]),
        "labels": torch.tensor([1]),
        "masks": torch.zeros(1, 64, 64, dtype=torch.uint8),
    }
    target["masks"][0, 12:42, 12:40] = 1

    losses = model.loss(image, [target])
    losses["loss"].backward()
    model.eval()
    with torch.no_grad():
        results = model.predict(image, score_threshold=0.0, max_detections=3)

    assert isinstance(model, EMCellFoundMaskRCNNInstanceSegmenter)
    assert torch.isfinite(losses["loss"])
    assert model.pre_nms_topk == 1000
    assert model.roi_batch_size_per_image == 64
    assert results[0]["boxes"].shape[1] == 4
    assert results[0]["masks"].shape[-2:] == (64, 64)


def test_mask2former_instance_segmenter_predict_and_loss_contract():
    model = get_model(
        MODEL_MASK2FORMER_INSTANCE,
        backbone_name="resnet34",
        img_size=64,
        num_classes=1,
        pretrained=False,
        neck_channels=16,
        head_channels=16,
        num_queries=8,
        mask_dim=16,
        transformer_heads=4,
        transformer_layers=1,
    )
    image = torch.zeros(1, 3, 64, 64)
    target = {
        "boxes": torch.tensor([[12.0, 12.0, 40.0, 42.0]]),
        "labels": torch.tensor([1]),
        "masks": torch.zeros(1, 64, 64, dtype=torch.uint8),
    }
    target["masks"][0, 12:42, 12:40] = 1

    losses = model.loss(image, [target])
    losses["loss"].backward()
    model.eval()
    with torch.no_grad():
        outputs = model(image)
        results = model.predict(image, score_threshold=0.0, max_detections=3)

    assert isinstance(model, EMCellFoundMask2FormerInstanceSegmenter)
    assert outputs["query_logits"].shape[1] == 8
    assert torch.isfinite(losses["loss"])
    assert results[0]["boxes"].shape[1] == 4
    assert results[0]["masks"].shape[-2:] == (64, 64)


class FakeInstanceModel(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()

    def predict(
        self,
        tensor,
        score_threshold=0.3,
        max_detections=100,
        nms_iou_threshold=0.5,
        mask_threshold=0.5,
    ):
        return [
            {
                "boxes": torch.tensor([[1.0, 2.0, 8.0, 9.0]], device=tensor.device),
                "scores": torch.tensor([0.75], device=tensor.device),
                "labels": torch.tensor([1], device=tensor.device),
                "masks": torch.ones(1, tensor.shape[-2], tensor.shape[-1], device=tensor.device),
            }
        ]


class FakeTrainingInstanceModel(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(1.0))

    def loss(self, images, targets):
        loss = self.weight * 0.0 + images.sum() * 0.0 + 1.0
        return {
            "loss": loss,
            "loss_cls": loss,
            "loss_obj": loss,
            "loss_box": loss,
            "loss_mask": loss,
            "num_pos": loss,
        }

    def predict(
        self,
        images,
        score_threshold=0.3,
        max_detections=100,
        nms_iou_threshold=0.5,
        mask_threshold=0.5,
    ):
        predictions = []
        for image in images:
            _, height, width = image.shape
            mask = torch.zeros(1, height, width, dtype=torch.bool, device=image.device)
            mask[:, 3:9, 2:10] = True
            predictions.append(
                {
                    "boxes": torch.tensor([[2.0, 3.0, 10.0, 9.0]], device=image.device),
                    "scores": torch.tensor([0.95], device=image.device),
                    "labels": torch.tensor([1], device=image.device),
                    "masks": mask,
                }
            )
        return predictions


def test_instance_segmentation_inference_task_single_image(monkeypatch):
    monkeypatch.setattr(ist, "EMCellFoundRTMInstanceSegmenter", FakeInstanceModel)

    result = run_instance_segmentation_inference_task(
        InstanceSegmentationInferenceRequest(
            checkpoint_path=None,
            backbone_name="resnet34",
            img_size=32,
            num_classes=2,
            image=np.zeros((16, 16, 3), dtype=np.uint8),
            device="cpu",
        )
    )

    assert result["boxes"].shape == (1, 4)
    assert result["scores"].tolist() == [0.75]
    assert result["masks"].shape == (1, 32, 32)


def test_prediction_to_instance_mask_uses_score_order():
    prediction = {
        "masks": torch.tensor(
            [
                [[True, True], [False, False]],
                [[False, True], [True, True]],
            ]
        ),
        "scores": torch.tensor([0.2, 0.9]),
    }

    mask = prediction_to_instance_mask(prediction)

    assert mask.dtype == np.uint16
    assert mask.tolist() == [[1, 2], [2, 2]]


def test_prediction_to_instance_mask_keeps_empty_mask_size():
    prediction = {
        "masks": torch.empty(0, 16, 12, dtype=torch.bool),
        "scores": torch.empty(0),
    }

    mask = prediction_to_instance_mask(prediction)

    assert mask.shape == (16, 12)
    assert mask.sum() == 0


def test_instance_segmentation_inference_task_folder_writes_csv(monkeypatch, tmp_path):
    monkeypatch.setattr(ist, "EMCellFoundRTMInstanceSegmenter", FakeInstanceModel)
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    Image.fromarray(np.zeros((16, 16, 3), dtype=np.uint8)).save(image_dir / "a.png")

    rows = run_instance_segmentation_inference_task(
        InstanceSegmentationInferenceRequest(
            checkpoint_path=None,
            backbone_name="resnet34",
            img_size=32,
            num_classes=2,
            image_folder=str(image_dir),
            output_csv=str(tmp_path / "preds.csv"),
            mask_output_folder=str(tmp_path / "instance_masks"),
            binary_mask_output_folder=str(tmp_path / "binary_masks"),
            device="cpu",
        )
    )

    assert len(rows) == 1
    assert rows[0]["label"] == 1
    assert rows[0]["score"] == 0.75
    assert (tmp_path / "preds.csv").exists()
    assert (tmp_path / "instance_masks" / "a_instances.png").exists()
    assert (tmp_path / "binary_masks" / "a_instance_0001.png").exists()


def test_instance_detection_metrics_for_perfect_prediction():
    mask = torch.zeros(1, 16, 16, dtype=torch.bool)
    mask[:, 3:9, 2:10] = True
    prediction = {
        "boxes": torch.tensor([[2.0, 3.0, 10.0, 9.0]]),
        "scores": torch.tensor([0.95]),
        "labels": torch.tensor([1]),
        "masks": mask,
    }
    target = {
        "boxes": torch.tensor([[2.0, 3.0, 10.0, 9.0]]),
        "labels": torch.tensor([1]),
        "masks": mask.to(torch.uint8),
    }

    metrics = ist._compute_instance_detection_metrics(
        [prediction],
        [target],
        prefix="val_",
    )

    assert metrics["val_mAP"] == 1.0
    assert metrics["val_AP50"] == 1.0
    assert metrics["val_AP75"] == 1.0
    assert metrics["val_mask_IoU"] == 1.0
    assert metrics["val_box_IoU"] == 1.0
    assert metrics["val_precision"] == 1.0
    assert metrics["val_recall"] == 1.0


def test_instance_segmentation_training_task_saves_checkpoint(monkeypatch, tmp_path):
    monkeypatch.setenv("EMCFSYS_MODEL_REGISTRY", str(tmp_path / "registry.json"))
    image_dir, annotation_path = _write_tiny_coco_dataset(tmp_path / "dataset")
    save_dir = tmp_path / "save"

    logs = run_instance_segmentation_training_task(
        InstanceSegmentationTrainingRequest(
            image_dir=str(image_dir),
            annotation_path=str(annotation_path),
            save_path=str(save_dir),
            backbone_name="resnet34",
            img_size=64,
            batch_size=1,
            epochs=1,
            lr=1e-4,
            device="cpu",
            pretrained=False,
            num_workers=0,
        )
    )

    assert logs
    assert (save_dir / "final_instance_segmentation.pth").exists()
    assert list(save_dir.glob("best_instance_segmentation_epoch*_Loss=*.pth"))
    assert (save_dir / "config.json").exists()
    assert (save_dir / "training_log.csv").exists()
    assert (save_dir / "metrics.json").exists()
    assert (tmp_path / "registry.json").exists()


def test_instance_segmentation_training_uses_separate_val_and_test_sets(
    monkeypatch,
    tmp_path,
):
    monkeypatch.setenv("EMCFSYS_MODEL_REGISTRY", str(tmp_path / "registry.json"))
    monkeypatch.setattr(ist, "EMCellFoundRTMInstanceSegmenter", FakeTrainingInstanceModel)
    train_image_dir, train_annotation_path = _write_tiny_coco_dataset(tmp_path / "train")
    val_image_dir, val_annotation_path = _write_tiny_coco_dataset(tmp_path / "val")
    test_image_dir, test_annotation_path = _write_tiny_coco_dataset(tmp_path / "test")
    save_dir = tmp_path / "save"
    messages = []

    logs = run_instance_segmentation_training_task(
        InstanceSegmentationTrainingRequest(
            image_dir=str(train_image_dir),
            annotation_path=str(train_annotation_path),
            save_path=str(save_dir),
            backbone_name="resnet34",
            img_size=16,
            batch_size=1,
            epochs=1,
            lr=1e-4,
            device="cpu",
            pretrained=False,
            num_workers=0,
            val_image_dir=str(val_image_dir),
            val_annotation_path=str(val_annotation_path),
            test_image_dir=str(test_image_dir),
            test_annotation_path=str(test_annotation_path),
        ),
        log=messages.append,
    )

    assert (save_dir / "final_instance_segmentation.pth").exists()
    assert any(item[-1].get("val_loss") == 1.0 for item in logs if isinstance(item[-1], dict))
    assert any(item[-1].get("test_loss") == 1.0 for item in logs if isinstance(item[-1], dict))
    assert any(item[-1].get("val_mAP") == 0.6 for item in logs if isinstance(item[-1], dict))
    assert any(item[-1].get("val_AP50") == 1.0 for item in logs if isinstance(item[-1], dict))
    assert any(item[-1].get("val_AP75") == 1.0 for item in logs if isinstance(item[-1], dict))
    assert any(item[-1].get("val_mask_IoU", 0.0) > 0.0 for item in logs if isinstance(item[-1], dict))
    assert any(item[-1].get("val_box_IoU", 0.0) > 0.0 for item in logs if isinstance(item[-1], dict))
    assert any(item[-1].get("val_precision") == 1.0 for item in logs if isinstance(item[-1], dict))
    assert any(item[-1].get("val_recall") == 1.0 for item in logs if isinstance(item[-1], dict))
    assert any(item[-1].get("test_mAP") == 0.6 for item in logs if isinstance(item[-1], dict))
    assert any("Final test loss 1.0000" in message for message in messages)
    assert any("test mAP 0.6000" in message for message in messages)


def test_instance_segmentation_training_skips_empty_val_and_test_sets(
    monkeypatch,
    tmp_path,
):
    monkeypatch.setenv("EMCFSYS_MODEL_REGISTRY", str(tmp_path / "registry.json"))
    monkeypatch.setattr(ist, "EMCellFoundRTMInstanceSegmenter", FakeTrainingInstanceModel)
    image_dir, annotation_path = _write_tiny_coco_dataset(tmp_path / "train")

    logs = run_instance_segmentation_training_task(
        InstanceSegmentationTrainingRequest(
            image_dir=str(image_dir),
            annotation_path=str(annotation_path),
            save_path=str(tmp_path / "save"),
            backbone_name="resnet34",
            img_size=16,
            batch_size=1,
            epochs=1,
            lr=1e-4,
            device="cpu",
            pretrained=False,
            num_workers=0,
            val_split=0.0,
        )
    )

    metric_logs = [item[-1] for item in logs if isinstance(item[-1], dict)]
    assert any(metrics.get("val_loss") is None for metrics in metric_logs)
    assert not any("test_loss" in metrics for metrics in metric_logs)


def test_instance_segmentation_training_can_use_only_val_json(
    monkeypatch,
    tmp_path,
):
    monkeypatch.setenv("EMCFSYS_MODEL_REGISTRY", str(tmp_path / "registry.json"))
    monkeypatch.setattr(ist, "EMCellFoundRTMInstanceSegmenter", FakeTrainingInstanceModel)
    image_dir, train_annotation_path = _write_tiny_coco_dataset(tmp_path / "train")
    _, val_annotation_path = _write_tiny_coco_dataset(tmp_path / "val_json_only")

    logs = run_instance_segmentation_training_task(
        InstanceSegmentationTrainingRequest(
            image_dir=str(image_dir),
            annotation_path=str(train_annotation_path),
            save_path=str(tmp_path / "save"),
            backbone_name="resnet34",
            img_size=16,
            batch_size=1,
            epochs=1,
            lr=1e-4,
            device="cpu",
            pretrained=False,
            num_workers=0,
            val_annotation_path=str(val_annotation_path),
        )
    )

    assert any(item[-1].get("val_loss") == 1.0 for item in logs if isinstance(item[-1], dict))
    assert any(item[-1].get("val_AP50") == 1.0 for item in logs if isinstance(item[-1], dict))
    assert not any(
        "test_loss" in item[-1]
        for item in logs
        if isinstance(item[-1], dict)
    )


def test_instance_segmentation_training_can_use_only_test_json(
    monkeypatch,
    tmp_path,
):
    monkeypatch.setenv("EMCFSYS_MODEL_REGISTRY", str(tmp_path / "registry.json"))
    monkeypatch.setattr(ist, "EMCellFoundRTMInstanceSegmenter", FakeTrainingInstanceModel)
    image_dir, train_annotation_path = _write_tiny_coco_dataset(tmp_path / "train")
    _, test_annotation_path = _write_tiny_coco_dataset(tmp_path / "test_json_only")

    logs = run_instance_segmentation_training_task(
        InstanceSegmentationTrainingRequest(
            image_dir=str(image_dir),
            annotation_path=str(train_annotation_path),
            save_path=str(tmp_path / "save"),
            backbone_name="resnet34",
            img_size=16,
            batch_size=1,
            epochs=1,
            lr=1e-4,
            device="cpu",
            pretrained=False,
            num_workers=0,
            test_annotation_path=str(test_annotation_path),
        )
    )

    assert any(item[-1].get("val_loss") is None for item in logs if isinstance(item[-1], dict))
    assert any(item[-1].get("test_loss") == 1.0 for item in logs if isinstance(item[-1], dict))
    assert any(item[-1].get("test_AP75") == 1.0 for item in logs if isinstance(item[-1], dict))


def test_instance_segmentation_training_requires_json_when_eval_image_dir_is_set(
    monkeypatch,
    tmp_path,
):
    monkeypatch.setenv("EMCFSYS_MODEL_REGISTRY", str(tmp_path / "registry.json"))
    monkeypatch.setattr(ist, "EMCellFoundRTMInstanceSegmenter", FakeTrainingInstanceModel)
    image_dir, annotation_path = _write_tiny_coco_dataset(tmp_path / "train")

    try:
        run_instance_segmentation_training_task(
            InstanceSegmentationTrainingRequest(
                image_dir=str(image_dir),
                annotation_path=str(annotation_path),
                save_path=str(tmp_path / "save"),
                backbone_name="resnet34",
                img_size=16,
                batch_size=1,
                epochs=1,
                lr=1e-4,
                device="cpu",
                pretrained=False,
                num_workers=0,
                val_image_dir=str(image_dir),
            )
        )
    except ValueError as error:
        assert "Validation COCO JSON is required" in str(error)
    else:
        raise AssertionError("Expected ValueError when validation image folder has no JSON")


def test_instance_segmentation_training_iterator_yields_realtime_logs(
    monkeypatch,
    tmp_path,
):
    monkeypatch.setenv("EMCFSYS_MODEL_REGISTRY", str(tmp_path / "registry.json"))
    monkeypatch.setattr(ist, "EMCellFoundRTMInstanceSegmenter", FakeTrainingInstanceModel)
    image_dir, annotation_path = _write_tiny_coco_dataset(tmp_path / "dataset")
    save_dir = tmp_path / "save"
    worker = iter_instance_segmentation_training_task(
        InstanceSegmentationTrainingRequest(
            image_dir=str(image_dir),
            annotation_path=str(annotation_path),
            save_path=str(save_dir),
            backbone_name="resnet34",
            img_size=16,
            batch_size=1,
            epochs=1,
            lr=1e-4,
            device="cpu",
            pretrained=False,
            num_workers=0,
        )
    )

    first_message = next(worker)
    remaining_messages = []
    try:
        while True:
            remaining_messages.append(next(worker))
    except StopIteration as stopped:
        logs = stopped.value

    assert first_message.startswith("Epoch 1 batch 1/1 loss")
    assert any("Epoch 1 finished" in message for message in remaining_messages)
    assert any("Final instance segmentation model saved" in message for message in remaining_messages)
    assert logs


def test_instance_segmentation_checkpoint_payload_saves_advanced_loss_weights(tmp_path):
    request = InstanceSegmentationTrainingRequest(
        image_dir=str(tmp_path / "images"),
        annotation_path=str(tmp_path / "instances.json"),
        save_path=str(tmp_path / "save"),
        use_advanced_mask_losses=True,
        boundary_loss_weight=0.2,
        focal_mask_loss_weight=0.3,
        tversky_loss_weight=0.4,
    )
    model = EMCellFoundRTMInstanceSegmenter(
        backbone_name="resnet34",
        num_classes=1,
        img_size=64,
        pretrained=False,
        neck_channels=16,
        head_channels=16,
        num_prototypes=8,
        boundary_loss_weight=0.2,
        focal_mask_loss_weight=0.3,
        tversky_loss_weight=0.4,
    )

    payload = ist._checkpoint_payload(
        request=request,
        model=model,
        class_names=["mitochondria"],
        epoch=1,
        loss=1.0,
    )

    assert payload["boundary_loss_weight"] == 0.2
    assert payload["focal_mask_loss_weight"] == 0.3
    assert payload["tversky_loss_weight"] == 0.4


def test_instance_segmentation_inference_uses_checkpoint_metadata(tmp_path):
    checkpoint_path = tmp_path / "checkpoint.pth"
    model = EMCellFoundRTMInstanceSegmenter(
        backbone_name="resnet34",
        num_classes=1,
        img_size=64,
        pretrained=False,
        neck_channels=16,
        head_channels=16,
        num_prototypes=8,
    )
    torch.save(
        {
            "task": "instance_segmentation",
            "model_name": "rtm_instance",
            "backbone_name": "resnet34",
            "img_size": 64,
            "num_classes": 1,
            "class_names": ["mitochondria"],
            "neck_channels": 16,
            "head_channels": 16,
            "num_prototypes": 8,
            "state_dict": model.state_dict(),
        },
        checkpoint_path,
    )

    result = run_instance_segmentation_inference_task(
        InstanceSegmentationInferenceRequest(
            checkpoint_path=str(checkpoint_path),
            backbone_name="emcellfound_vit_base",
            img_size=32,
            num_classes=99,
            image=np.zeros((64, 64, 3), dtype=np.uint8),
            device="cpu",
            score_threshold=1.0,
        )
    )

    assert result["masks"].shape[-2:] == (64, 64)
