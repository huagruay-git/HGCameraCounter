#!/usr/bin/env python3
"""Validate YOLO dataset labels and image/label consistency."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Tuple

import yaml


IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def collect_images(dir_path: Path) -> List[Path]:
    if not dir_path.exists():
        return []
    out = []
    for p in dir_path.rglob("*"):
        if p.is_file() and p.suffix.lower() in IMG_EXTS:
            out.append(p)
    return sorted(out)


def validate_label_file(path: Path, num_classes: int) -> Tuple[int, int, List[str]]:
    errors: List[str] = []
    lines = 0
    valid = 0
    if not path.exists():
        return 0, 0, [f"missing label: {path}"]

    with path.open("r", encoding="utf-8") as f:
        for idx, raw in enumerate(f, start=1):
            s = raw.strip()
            if not s:
                continue
            lines += 1
            parts = s.split()
            if len(parts) != 5:
                errors.append(f"{path}:{idx} need 5 fields, got {len(parts)}")
                continue
            try:
                cid = int(parts[0])
                vals = [float(x) for x in parts[1:]]
            except ValueError:
                errors.append(f"{path}:{idx} parse error")
                continue
            if cid < 0 or cid >= num_classes:
                errors.append(f"{path}:{idx} class_id {cid} out of range [0,{num_classes-1}]")
                continue
            if any(v < 0.0 or v > 1.0 for v in vals):
                errors.append(f"{path}:{idx} normalized values must be in [0,1]")
                continue
            if vals[2] <= 0.0 or vals[3] <= 0.0:
                errors.append(f"{path}:{idx} width/height must be >0")
                continue
            valid += 1

    return lines, valid, errors


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Validate YOLO label files")
    p.add_argument("--dataset", default="data/yolo_head_dataset", help="Dataset root")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(args.dataset)
    data_yaml = root / "dataset.yaml"
    if not data_yaml.exists():
        raise FileNotFoundError(f"dataset.yaml not found: {data_yaml}")

    cfg = yaml.safe_load(data_yaml.read_text(encoding="utf-8"))
    names = cfg.get("names", {})
    num_classes = len(names)
    if num_classes <= 0:
        raise ValueError("No classes found in dataset.yaml names")

    train_imgs = collect_images(root / "images" / "train")
    val_imgs = collect_images(root / "images" / "val")
    all_imgs = train_imgs + val_imgs

    total_lines = 0
    total_valid = 0
    all_errors: List[str] = []

    for img in all_imgs:
        split = "train" if "images/train" in str(img).replace("\\", "/") else "val"
        label = root / "labels" / split / (img.stem + ".txt")
        lines, valid, errors = validate_label_file(label, num_classes)
        total_lines += lines
        total_valid += valid
        all_errors.extend(errors)

    print(f"images_train={len(train_imgs)} images_val={len(val_imgs)}")
    print(f"labels_checked={len(all_imgs)} lines={total_lines} valid={total_valid} errors={len(all_errors)}")

    if len(train_imgs) == 0:
        all_errors.append("no images found in images/train")
    if len(val_imgs) == 0:
        print("WARN: images/val is empty (train script will fallback to train as val).")
    if len(all_imgs) == 0:
        all_errors.append("dataset has zero images (train+val)")

    if all_errors:
        for e in all_errors[:200]:
            print(f"ERROR: {e}")
        raise SystemExit(1)

    print("OK")


if __name__ == "__main__":
    main()
