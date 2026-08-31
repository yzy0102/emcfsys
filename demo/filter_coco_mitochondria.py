"""Keep only the Mitochondria category in COCO instance JSON files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


SPLIT_FILES = ("train.json", "val.json", "test.json")


def filter_split(path: Path) -> tuple[int, int, int]:
    coco = json.loads(path.read_text(encoding="utf-8"))
    annotations = [
        annotation
        for annotation in coco.get("annotations", [])
        if int(annotation.get("category_id", -1)) == 1
    ]
    removed = len(coco.get("annotations", [])) - len(annotations)

    # Keep all images because every split image already has Mitochondria labels.
    for annotation_id, annotation in enumerate(annotations, start=1):
        annotation["id"] = annotation_id

    coco["annotations"] = annotations
    coco["categories"] = [
        {"id": 1, "name": "Mitochondria", "supercategory": "Mitochondria"}
    ]
    path.write_text(json.dumps(coco, ensure_ascii=False, indent=2), encoding="utf-8")
    return len(coco.get("images", [])), len(annotations), removed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=Path("datasets_temp/MitoInstanceSegDataset"),
    )
    args = parser.parse_args()

    for filename in SPLIT_FILES:
        path = args.dataset_dir / filename
        if not path.is_file():
            raise FileNotFoundError(f"Missing COCO split file: {path}")
        images, annotations, removed = filter_split(path)
        print(
            f"{filename}: images={images}, mitochondria_annotations={annotations}, "
            f"removed_category_2={removed}"
        )


if __name__ == "__main__":
    main()
