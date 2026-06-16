#!/usr/bin/env python3
"""Train custom salon YOLO model and optionally mine hard negatives."""

from __future__ import annotations

import argparse
import csv
import shutil
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import yaml


IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff", ".jp2"}
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


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


def _load_data_cfg(path: Path) -> Dict:
    cfg = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(cfg, dict):
        raise ValueError("data.yaml must be a dict")
    return cfg


def _resolve_dataset_dirs(data_yaml: Path) -> Tuple[Path, Path, Path]:
    cfg = _load_data_cfg(data_yaml)
    base = data_yaml.parent.resolve()
    root = Path(cfg.get("path", "."))
    if not root.is_absolute():
        root = (base / root).resolve()
    train_dir = (root / str(cfg.get("train", "images/train"))).resolve()
    val_dir = (root / str(cfg.get("val", "images/val"))).resolve()
    label_root = (root / "labels").resolve()
    return train_dir, val_dir, label_root


def _load_augment_cfg(path: Path) -> Dict:
    if not path.exists():
        return {}
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        return {}
    return payload


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train salon YOLO model")
    p.add_argument("--data", default="data/yolo_salon_dataset/data.yaml")
    p.add_argument("--model", default="yolov8m.pt", help="yolov8/YOLO11 checkpoint for transfer learning")
    p.add_argument("--epochs", type=int, default=120)
    p.add_argument("--imgsz", type=int, default=960)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--device", default="auto")
    p.add_argument("--patience", type=int, default=30)
    p.add_argument("--project", default="runs/train")
    p.add_argument("--name", default="yolo_salon_custom")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--freeze", type=int, default=10, help="Freeze first N layers for transfer learning warmup")
    p.add_argument("--augment-config", default="data/yolo_salon_dataset/augment.yaml")
    p.add_argument("--exist-ok", action="store_true")
    p.add_argument("--save-period", type=int, default=10)
    p.add_argument("--mine-hard-negatives", action="store_true")
    p.add_argument("--hard-negative-out", default="data/performance_feedback/hard_negative_mined")
    return p.parse_args()


def _label_file_for_image(label_root: Path, img_path: Path, split: str) -> Path:
    return (label_root / split / f"{img_path.stem}.txt").resolve()


def _get_gt_class_id(label_file: Path) -> int:
    if not label_file.exists():
        return -1
    text = label_file.read_text(encoding="utf-8").strip()
    if not text:
        return -1
    first = text.splitlines()[0].strip().split()
    if not first:
        return -1
    try:
        return int(first[0])
    except Exception:
        return -1


def _iter_images(root: Path) -> List[Path]:
    if not root.exists():
        return []
    return sorted([p for p in root.iterdir() if p.is_file() and p.suffix.lower() in IMG_EXTS])


def _mine_hard_negatives(weights: Path, data_yaml: Path, output_dir: Path, device: str, imgsz: int = 960) -> int:
    from ultralytics import YOLO

    output_dir.mkdir(parents=True, exist_ok=True)
    model = YOLO(str(weights))
    train_dir, val_dir, label_root = _resolve_dataset_dirs(data_yaml)
    val_images = _iter_images(val_dir)
    if not val_images:
        return 0

    cfg = _load_data_cfg(data_yaml)
    names_map = cfg.get("names", {})
    if isinstance(names_map, list):
        names = {i: str(v) for i, v in enumerate(names_map)}
    elif isinstance(names_map, dict):
        names = {int(k): str(v) for k, v in names_map.items()}
    else:
        names = {}

    rows: List[Dict[str, str]] = []
    copied = 0
    for img_path in val_images:
        gt_id = _get_gt_class_id(_label_file_for_image(label_root, img_path, "val"))
        gt_name = names.get(gt_id, "background") if gt_id >= 0 else "background"

        result = model.predict(source=str(img_path), imgsz=imgsz, device=device, conf=0.20, iou=0.50, verbose=False)[0]
        boxes = getattr(result, "boxes", None)
        if boxes is None or boxes.cls is None or len(boxes.cls) == 0:
            pred_id = -1
            pred_conf = 0.0
            pred_name = "none"
        else:
            confs = boxes.conf.float().cpu().tolist()
            ids = boxes.cls.int().cpu().tolist()
            best_idx = max(range(len(confs)), key=lambda i: confs[i])
            pred_id = int(ids[best_idx])
            pred_conf = float(confs[best_idx])
            pred_name = str(names.get(pred_id, f"class_{pred_id}"))

        mismatch = pred_id != gt_id
        low_conf = pred_conf < 0.45
        if mismatch or low_conf:
            dst = output_dir / img_path.name
            shutil.copy2(img_path, dst)
            copied += 1
            rows.append(
                {
                    "image": img_path.name,
                    "gt_id": str(gt_id),
                    "gt_name": gt_name,
                    "pred_id": str(pred_id),
                    "pred_name": pred_name,
                    "pred_conf": f"{pred_conf:.4f}",
                    "reason": "mismatch" if mismatch else "low_conf",
                }
            )

    if rows:
        csv_path = output_dir / "hard_negative_report.csv"
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
    return copied


def main() -> int:
    from ultralytics import YOLO

    args = parse_args()
    root = PROJECT_ROOT
    data_yaml = (root / args.data).resolve() if not Path(args.data).is_absolute() else Path(args.data).resolve()
    augment_yaml = (
        (root / args.augment_config).resolve()
        if not Path(args.augment_config).is_absolute()
        else Path(args.augment_config).resolve()
    )
    if not data_yaml.exists():
        raise FileNotFoundError(f"data.yaml not found: {data_yaml}")

    train_dir, val_dir, _ = _resolve_dataset_dirs(data_yaml)
    if not train_dir.exists():
        raise FileNotFoundError(f"train dir not found: {train_dir}")

    device = _select_device(args.device)
    aug_cfg = _load_augment_cfg(augment_yaml)
    print(f"DEVICE_SELECTED {device}")
    print(f"TRAIN_DIR {train_dir}")
    print(f"VAL_DIR {val_dir}")

    model = YOLO(str(args.model))
    train_kwargs = {
        "data": str(data_yaml),
        "epochs": int(args.epochs),
        "imgsz": int(args.imgsz),
        "batch": int(args.batch),
        "workers": int(args.workers),
        "device": device,
        "patience": int(args.patience),
        "project": str(args.project),
        "name": str(args.name),
        "seed": int(args.seed),
        "freeze": int(args.freeze),
        "exist_ok": bool(args.exist_ok),
        "save_period": int(args.save_period),
        "cos_lr": True,
        "optimizer": "auto",
        "amp": True,
    }
    train_kwargs.update(aug_cfg)

    results = model.train(**train_kwargs)
    save_dir = Path(getattr(results, "save_dir", Path(args.project) / args.name))
    best_pt = (save_dir / "weights" / "best.pt").resolve()
    print(f"TRAIN_SAVE_DIR {save_dir}")
    print(f"BEST_MODEL {best_pt}")

    if args.mine_hard_negatives and best_pt.exists():
        out_dir = (root / args.hard_negative_out).resolve() if not Path(args.hard_negative_out).is_absolute() else Path(args.hard_negative_out).resolve()
        mined = _mine_hard_negatives(best_pt, data_yaml, out_dir, device=device, imgsz=int(args.imgsz))
        print(f"HARD_NEGATIVE_MINED {mined} -> {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
