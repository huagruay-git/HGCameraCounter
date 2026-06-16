#!/usr/bin/env python3
"""Train chair-zone service classifier (haircut vs not_haircut) from performance feedback folders."""

from __future__ import annotations

import argparse
import random
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import Iterable, List, Tuple

from ultralytics import YOLO


PROJECT_ROOT = Path(__file__).resolve().parent.parent
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff", ".jp2"}


def _list_images_recursive(root: Path) -> List[Path]:
    if not root.exists() or (not root.is_dir()):
        return []
    return sorted([p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in IMG_EXTS])


def _resolve_path(path_like: str) -> Path:
    p = Path(str(path_like or "")).expanduser()
    if p.is_absolute():
        return p
    return (PROJECT_ROOT / p).resolve()


def _safe_stem(name: str) -> str:
    txt = "".join(ch if (ch.isalnum() or ch in ("-", "_")) else "_" for ch in str(name or ""))
    return txt.strip("_") or "img"


def _split_train_val(items: List[Path], train_ratio: float, rng: random.Random) -> Tuple[List[Path], List[Path]]:
    if not items:
        return [], []
    shuffled = list(items)
    rng.shuffle(shuffled)
    if len(shuffled) <= 1:
        return shuffled, []
    train_n = int(round(len(shuffled) * train_ratio))
    train_n = max(1, min(len(shuffled) - 1, train_n))
    return shuffled[:train_n], shuffled[train_n:]


def _upsample_to_target(items: List[Path], target_n: int, rng: random.Random) -> List[Path]:
    if not items:
        return []
    out = list(items)
    while len(out) < target_n:
        out.append(rng.choice(items))
    return out


def _reset_dataset_root(dataset_root: Path) -> None:
    if dataset_root.exists():
        shutil.rmtree(dataset_root, ignore_errors=True)
    for split in ("train", "val"):
        for cls_name in ("haircut", "not_haircut"):
            (dataset_root / split / cls_name).mkdir(parents=True, exist_ok=True)


def _copy_samples(samples: Iterable[Path], dst_dir: Path, prefix: str) -> int:
    count = 0
    for idx, src in enumerate(samples):
        stem = _safe_stem(src.stem)
        dst = dst_dir / f"{prefix}_{idx:06d}_{stem}{src.suffix.lower()}"
        shutil.copy2(src, dst)
        count += 1
    return count


def _backup_if_exists(path: Path) -> None:
    if not path.exists():
        return
    backup_dir = path.parent / "backups"
    backup_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = backup_dir / f"{path.stem}_before_chair_cls_{stamp}{path.suffix}"
    shutil.copy2(path, backup_path)
    print(f"BACKUP_MODEL {path} -> {backup_path}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train chair haircut classifier from performance feedback folders")
    p.add_argument(
        "--positive-dir",
        default="data/performance_feedback/haircut",
        help="Folder for positive class (haircut)",
    )
    p.add_argument(
        "--negative-dirs",
        nargs="+",
        default=[
            "data/performance_feedback/no haircut",
            "data/performance_feedback/customerwash",
            "data/performance_feedback/no customerwash",
        ],
        help="Folders treated as negative class (not_haircut)",
    )
    p.add_argument("--dataset-root", default="data/chair_service_cls_dataset")
    p.add_argument("--output-model", default="models/chair_service_cls.pt")
    p.add_argument("--model", default="yolov8n-cls.pt", help="Classification base model")
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--imgsz", type=int, default=224)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--device", default="auto", help="auto|cpu|cuda|mps")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--patience", type=int, default=20)
    p.add_argument("--train-split", type=float, default=0.85)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--project", default="runs/classify")
    p.add_argument("--name", default="chair_service_cls")
    p.add_argument("--exist-ok", action="store_true")
    return p.parse_args()


def _select_device(raw_device: str) -> str:
    dev = str(raw_device or "auto").strip().lower()
    if dev != "auto":
        return dev
    try:
        import torch
        if torch.cuda.is_available():
            return "0"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
    except Exception:
        pass
    return "cpu"


def main() -> int:
    args = parse_args()
    rng = random.Random(int(args.seed))

    positive_dir = _resolve_path(args.positive_dir)
    negative_dirs = [_resolve_path(x) for x in (args.negative_dirs or [])]
    dataset_root = _resolve_path(args.dataset_root)
    output_model = _resolve_path(args.output_model)

    pos = _list_images_recursive(positive_dir)
    neg: List[Path] = []
    for d in negative_dirs:
        neg.extend(_list_images_recursive(d))
    neg = sorted(neg)

    print(f"SOURCE positive(haircut)={len(pos)} from {positive_dir}")
    for d in negative_dirs:
        print(f"SOURCE negative += {len(_list_images_recursive(d))} from {d}")
    print(f"SOURCE negative(total)={len(neg)}")

    if len(pos) <= 0:
        raise FileNotFoundError(f"No positive images found: {positive_dir}")
    if len(neg) <= 0:
        raise FileNotFoundError("No negative images found. Add images to no haircut / customerwash / no customerwash")

    train_ratio = min(0.99, max(0.01, float(args.train_split)))
    pos_train, pos_val = _split_train_val(pos, train_ratio, rng)
    neg_train, neg_val = _split_train_val(neg, train_ratio, rng)

    # Avoid train/val leakage: split first, then upsample only training split.
    train_target_n = max(len(pos_train), len(neg_train))
    pos_train_bal = _upsample_to_target(pos_train, train_target_n, rng)
    neg_train_bal = _upsample_to_target(neg_train, train_target_n, rng)
    rng.shuffle(pos_train_bal)
    rng.shuffle(neg_train_bal)

    _reset_dataset_root(dataset_root)
    n_pos_train = _copy_samples(pos_train_bal, dataset_root / "train" / "haircut", "haircut")
    n_pos_val = _copy_samples(pos_val, dataset_root / "val" / "haircut", "haircut")
    n_neg_train = _copy_samples(neg_train_bal, dataset_root / "train" / "not_haircut", "not_haircut")
    n_neg_val = _copy_samples(neg_val, dataset_root / "val" / "not_haircut", "not_haircut")
    print(
        "DATASET_READY "
        f"root={dataset_root} "
        f"train(haircut={n_pos_train},not_haircut={n_neg_train}) "
        f"val(haircut={n_pos_val},not_haircut={n_neg_val})"
    )

    model = YOLO(str(args.model))
    device = _select_device(args.device)
    print(f"DEVICE_SELECTED {device} (requested={args.device})")
    results = model.train(
        data=str(dataset_root),
        epochs=int(args.epochs),
        imgsz=int(args.imgsz),
        batch=int(args.batch),
        device=str(device),
        workers=int(args.workers),
        patience=int(args.patience),
        project=str(args.project),
        name=str(args.name),
        seed=int(args.seed),
        exist_ok=bool(args.exist_ok),
    )

    save_dir = Path(getattr(results, "save_dir", ""))
    best = (save_dir / "weights" / "best.pt").resolve()
    if not best.exists():
        fallback = (PROJECT_ROOT / str(args.project) / str(args.name) / "weights" / "best.pt").resolve()
        if fallback.exists():
            best = fallback
    if not best.exists():
        raise FileNotFoundError("Training completed but best.pt was not found")

    output_model.parent.mkdir(parents=True, exist_ok=True)
    _backup_if_exists(output_model)
    shutil.copy2(best, output_model)
    print(f"BEST_MODEL {best}")
    print(f"APPLY_MODEL {output_model}")
    print("DONE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
