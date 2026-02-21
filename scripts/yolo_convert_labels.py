#!/usr/bin/env python3
"""Convert custom/COCO annotations to YOLO txt labels."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import cv2


DEFAULT_CLASS_MAP = {
    "person": 0,
    "head_customer": 1,
    "head": 1,
    "staff_uniform": 2,
    "staff": 2,
}


def xyxy_to_yolo(x1: float, y1: float, x2: float, y2: float, w: float, h: float) -> Tuple[float, float, float, float]:
    x1 = max(0.0, min(x1, w))
    y1 = max(0.0, min(y1, h))
    x2 = max(0.0, min(x2, w))
    y2 = max(0.0, min(y2, h))
    bw = max(0.0, x2 - x1)
    bh = max(0.0, y2 - y1)
    cx = x1 + (bw / 2.0)
    cy = y1 + (bh / 2.0)
    if w <= 0 or h <= 0:
        return 0.0, 0.0, 0.0, 0.0
    return cx / w, cy / h, bw / w, bh / h


def detect_split(file_name: str, default_split: str = "train") -> str:
    p = file_name.replace("\\", "/").lower()
    if "/val/" in p or p.startswith("val/"):
        return "val"
    if "/train/" in p or p.startswith("train/"):
        return "train"
    return default_split


def ensure_image_shape(img_path: Path, width: int | None, height: int | None) -> Tuple[int, int]:
    if width and height and width > 0 and height > 0:
        return width, height
    img = cv2.imread(str(img_path))
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {img_path}")
    h, w = img.shape[:2]
    return int(w), int(h)


def write_label_file(label_path: Path, rows: List[Tuple[int, float, float, float, float]]) -> None:
    label_path.parent.mkdir(parents=True, exist_ok=True)
    with label_path.open("w", encoding="utf-8") as f:
        for cid, cx, cy, bw, bh in rows:
            if bw <= 0 or bh <= 0:
                continue
            f.write(f"{cid} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n")


def convert_custom_format(data: Dict, dataset_root: Path, class_map: Dict[str, int], default_split: str) -> Tuple[int, int]:
    converted = 0
    skipped = 0
    images = data.get("images", [])
    for item in images:
        file_name = str(item.get("file_name", "")).strip()
        if not file_name:
            skipped += 1
            continue

        img_path = dataset_root / file_name if not Path(file_name).is_absolute() else Path(file_name)
        split = detect_split(file_name, default_split=default_split)
        label_name = Path(file_name).with_suffix(".txt").name
        label_path = dataset_root / "labels" / split / label_name

        try:
            width, height = ensure_image_shape(img_path, item.get("width"), item.get("height"))
        except Exception:
            skipped += 1
            continue

        rows = []
        for ann in item.get("annotations", []):
            cname = str(ann.get("class", "")).strip().lower()
            if cname not in class_map:
                continue
            cid = class_map[cname]

            if "bbox_xyxy" in ann:
                x1, y1, x2, y2 = [float(v) for v in ann["bbox_xyxy"]]
            elif "bbox_xywh" in ann:
                x, y, bw, bh = [float(v) for v in ann["bbox_xywh"]]
                x1, y1, x2, y2 = x, y, x + bw, y + bh
            else:
                continue

            cx, cy, bw_n, bh_n = xyxy_to_yolo(x1, y1, x2, y2, float(width), float(height))
            rows.append((cid, cx, cy, bw_n, bh_n))

        write_label_file(label_path, rows)
        converted += 1

    return converted, skipped


def convert_coco_format(data: Dict, dataset_root: Path, class_map: Dict[str, int], default_split: str) -> Tuple[int, int]:
    converted = 0
    skipped = 0

    categories = {int(c["id"]): str(c["name"]).strip().lower() for c in data.get("categories", [])}
    image_map = {int(im["id"]): im for im in data.get("images", [])}
    anns_by_image: Dict[int, List[Dict]] = {}
    for ann in data.get("annotations", []):
        anns_by_image.setdefault(int(ann["image_id"]), []).append(ann)

    for image_id, im in image_map.items():
        file_name = str(im.get("file_name", "")).strip()
        if not file_name:
            skipped += 1
            continue

        width = int(im.get("width", 0))
        height = int(im.get("height", 0))
        img_path = dataset_root / file_name if not Path(file_name).is_absolute() else Path(file_name)
        try:
            width, height = ensure_image_shape(img_path, width, height)
        except Exception:
            skipped += 1
            continue

        split = detect_split(file_name, default_split=default_split)
        label_name = Path(file_name).with_suffix(".txt").name
        label_path = dataset_root / "labels" / split / label_name

        rows = []
        for ann in anns_by_image.get(image_id, []):
            cname = categories.get(int(ann.get("category_id", -1)), "")
            if cname not in class_map:
                continue
            cid = class_map[cname]

            bbox = ann.get("bbox", None)
            if not bbox or len(bbox) < 4:
                continue
            x, y, bw, bh = [float(v) for v in bbox[:4]]
            x1, y1, x2, y2 = x, y, x + bw, y + bh
            cx, cy, bw_n, bh_n = xyxy_to_yolo(x1, y1, x2, y2, float(width), float(height))
            rows.append((cid, cx, cy, bw_n, bh_n))

        write_label_file(label_path, rows)
        converted += 1

    return converted, skipped


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Convert annotations to YOLO txt format")
    p.add_argument("--input", required=True, help="Path to source JSON annotation file")
    p.add_argument(
        "--dataset-root",
        default="data/yolo_head_dataset",
        help="Dataset root containing images/ and labels/",
    )
    p.add_argument("--default-split", default="train", choices=["train", "val"], help="Fallback split")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    dataset_root = Path(args.dataset_root)

    with input_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, dict):
        raise ValueError("Input JSON must be an object")

    if "annotations" in data and "images" in data and "categories" in data:
        converted, skipped = convert_coco_format(data, dataset_root, DEFAULT_CLASS_MAP, args.default_split)
        fmt = "coco"
    elif "images" in data:
        converted, skipped = convert_custom_format(data, dataset_root, DEFAULT_CLASS_MAP, args.default_split)
        fmt = "custom"
    else:
        raise ValueError("Unsupported format. Need custom {images:[...]} or COCO keys")

    print(f"format={fmt} converted={converted} skipped={skipped}")


if __name__ == "__main__":
    main()
