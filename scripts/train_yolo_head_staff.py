#!/usr/bin/env python3
"""Train Ultralytics YOLO with salon classes (person/head_customer/staff_uniform).

If validation split is empty, this script falls back to using train split as val
for quick bootstrapping on small datasets.
"""

from __future__ import annotations

import argparse
import tempfile
from pathlib import Path

import yaml
from ultralytics import YOLO


IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff", ".jp2"}


def _resolve_dataset_yaml(data_path: str) -> Path:
    p = Path(data_path).expanduser()
    if p.is_absolute() and p.exists():
        return p
    if p.exists():
        return p.resolve()
    root = Path(__file__).resolve().parent.parent
    cand = root / data_path
    if cand.exists():
        return cand.resolve()
    return p


def _count_images(dir_path: Path) -> int:
    if not dir_path.exists():
        return 0
    return sum(1 for p in dir_path.rglob("*") if p.is_file() and p.suffix.lower() in IMG_EXTS)


def _select_device(raw_device: str) -> str:
    dev = str(raw_device or "auto").strip().lower()
    if dev != "auto":
        return dev
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
    except Exception:
        pass
    return "cpu"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train YOLO for salon head/staff classes")
    p.add_argument("--data", default="data/yolo_head_dataset/dataset.yaml", help="Path to dataset.yaml")
    p.add_argument("--model", default="yolov8n.pt", help="Base model (e.g. yolov8n.pt)")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--device", default="auto", help="auto|cpu|mps|cuda")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--patience", type=int, default=30)
    p.add_argument("--project", default="runs/train")
    p.add_argument("--name", default="yolo_head_staff")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--exist-ok", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    data_path = _resolve_dataset_yaml(args.data)
    if not data_path.exists():
        raise FileNotFoundError(f"dataset.yaml not found: {data_path}")

    cfg = yaml.safe_load(data_path.read_text(encoding="utf-8")) or {}
    base = data_path.parent.resolve()
    project_root = Path(__file__).resolve().parent.parent
    path_cfg = Path(cfg.get("path", "."))
    if path_cfg.is_absolute():
        path_root = path_cfg
    else:
        candidates = [
            (base / path_cfg).resolve(),
            (project_root / path_cfg).resolve(),
            base,
        ]
        path_root = next((c for c in candidates if c.exists()), candidates[0])
    train_rel = str(cfg.get("train", "images/train"))
    val_rel = str(cfg.get("val", "images/val"))
    train_dir = (path_root / train_rel).resolve()
    val_dir = (path_root / val_rel).resolve()

    train_count = _count_images(train_dir)
    val_count = _count_images(val_dir)
    print(f"DATASET_CHECK train_dir={train_dir} images={train_count}")
    print(f"DATASET_CHECK val_dir={val_dir} images={val_count}")
    if train_count <= 0:
        raise FileNotFoundError(
            f"No training images found in {train_dir}. "
            "Add images first in Dataset Lab or put files under images/train."
        )

    train_data_arg = str(data_path)
    tmp_yaml_path: Path | None = None
    if val_count <= 0:
        # For small bootstrap datasets, allow val to reuse train.
        cfg["val"] = cfg.get("train", "images/train")
        fd, tmp_name = tempfile.mkstemp(prefix="dataset_bootstrap_", suffix=".yaml")
        Path(tmp_name).write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
        tmp_yaml_path = Path(tmp_name)
        train_data_arg = str(tmp_yaml_path)
        print(f"DATASET_NOTE val split is empty; temporarily using train as val ({cfg['val']})")

    device = _select_device(args.device)
    print(f"DEVICE_SELECTED: {device} (requested={args.device})")
    model = YOLO(args.model)
    try:
        results = model.train(
            data=train_data_arg,
            epochs=args.epochs,
            imgsz=args.imgsz,
            batch=args.batch,
            device=device,
            workers=args.workers,
            patience=args.patience,
            project=args.project,
            name=args.name,
            seed=args.seed,
            exist_ok=args.exist_ok,
        )
    finally:
        if tmp_yaml_path is not None:
            try:
                tmp_yaml_path.unlink(missing_ok=True)
            except Exception:
                pass

    best_pt = None
    save_dir = getattr(results, "save_dir", None)
    if save_dir is not None:
        cand = Path(save_dir) / "weights" / "best.pt"
        if cand.exists():
            best_pt = cand

    print(f"train_done save_dir={save_dir}")
    if best_pt:
        print(f"best_model={best_pt}")


if __name__ == "__main__":
    main()
