#!/usr/bin/env python3
"""Prepare balanced YOLO dataset for salon classes.

Classes:
0 = customer_haircut
1 = staff_barber
2 = customer_wash
"""

from __future__ import annotations

import argparse
import random
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import yaml


IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff", ".jp2"}
CLASS_TO_ID = {
    "customer_haircut": 0,
    "staff_barber": 1,
    "customer_wash": 2,
}


PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


@dataclass
class Sample:
    path: Path
    class_name: str
    is_oversampled: bool = False


def _select_device(raw: str) -> str:
    req = str(raw or "auto").strip().lower()
    if req != "auto":
        return req
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
    except Exception:
        pass
    return "cpu"


def _list_images(root: Path) -> List[Path]:
    if not root.exists():
        return []
    return sorted([p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in IMG_EXTS])


def _safe_stem(name: str) -> str:
    txt = "".join(ch if (ch.isalnum() or ch in ("-", "_")) else "_" for ch in name)
    txt = txt.strip("_")
    return txt or "img"


def _make_balanced(samples_by_class: Dict[str, List[Path]], seed: int) -> Dict[str, List[Sample]]:
    rng = random.Random(int(seed))
    non_empty = [v for v in samples_by_class.values() if v]
    if not non_empty:
        return {k: [] for k in samples_by_class}
    target = max(len(v) for v in non_empty)

    out: Dict[str, List[Sample]] = {}
    for cls_name, items in samples_by_class.items():
        entries = [Sample(path=p, class_name=cls_name, is_oversampled=False) for p in items]
        if not entries:
            out[cls_name] = entries
            continue
        while len(entries) < target:
            src = rng.choice(items)
            entries.append(Sample(path=src, class_name=cls_name, is_oversampled=True))
        out[cls_name] = entries
    return out


def _split(samples: List[Sample], val_ratio: float, seed: int) -> Tuple[List[Sample], List[Sample]]:
    if not samples:
        return [], []
    rng = random.Random(int(seed))
    arr = list(samples)
    rng.shuffle(arr)
    val_n = max(1, int(round(len(arr) * float(val_ratio)))) if len(arr) > 1 else 0
    val_n = min(max(0, val_n), len(arr))
    val = arr[:val_n]
    train = arr[val_n:]
    if not train and val:
        train = [val.pop()]
    return train, val


def _prepare_root(dataset_root: Path) -> None:
    if dataset_root.exists():
        shutil.rmtree(dataset_root, ignore_errors=True)
    for split in ("train", "val"):
        (dataset_root / "images" / split).mkdir(parents=True, exist_ok=True)
        (dataset_root / "labels" / split).mkdir(parents=True, exist_ok=True)


def _predict_person_box(yolo_model, image_path: Path, device: str, imgsz: int) -> Tuple[float, float, float, float]:
    import cv2

    img = cv2.imread(str(image_path))
    if img is None:
        return (0.5, 0.5, 1.0, 1.0)
    h, w = img.shape[:2]
    if h <= 1 or w <= 1:
        return (0.5, 0.5, 1.0, 1.0)
    try:
        result = yolo_model.predict(
            source=img,
            classes=[0],
            conf=0.15,
            iou=0.50,
            imgsz=int(imgsz),
            device=device,
            verbose=False,
        )[0]
        boxes = getattr(result, "boxes", None)
        if boxes is None or boxes.xyxy is None or len(boxes.xyxy) == 0:
            return (0.5, 0.5, 1.0, 1.0)
        xyxy = boxes.xyxy.cpu().numpy()
        best = None
        best_area = -1.0
        for row in xyxy:
            x1, y1, x2, y2 = [float(v) for v in row.tolist()]
            area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
            if area > best_area:
                best_area = area
                best = (x1, y1, x2, y2)
        if best is None:
            return (0.5, 0.5, 1.0, 1.0)
        x1, y1, x2, y2 = best
        cx = ((x1 + x2) * 0.5) / w
        cy = ((y1 + y2) * 0.5) / h
        bw = (x2 - x1) / w
        bh = (y2 - y1) / h
        cx = min(max(cx, 0.0), 1.0)
        cy = min(max(cy, 0.0), 1.0)
        bw = min(max(bw, 1.0 / w), 1.0)
        bh = min(max(bh, 1.0 / h), 1.0)
        return (cx, cy, bw, bh)
    except Exception:
        return (0.5, 0.5, 1.0, 1.0)


def _copy_labeled_sample(
    sample: Sample,
    split: str,
    index: int,
    dataset_root: Path,
    yolo_model,
    device: str,
    imgsz: int,
) -> None:
    stem = _safe_stem(sample.path.stem)
    over = "os" if sample.is_oversampled else "raw"
    base_name = f"{sample.class_name}_{index:07d}_{over}_{stem}{sample.path.suffix.lower()}"
    dst_img = dataset_root / "images" / split / base_name
    dst_lbl = dataset_root / "labels" / split / f"{Path(base_name).stem}.txt"
    shutil.copy2(sample.path, dst_img)

    cls_id = CLASS_TO_ID[sample.class_name]
    cx, cy, bw, bh = _predict_person_box(yolo_model, dst_img, device=device, imgsz=imgsz)
    dst_lbl.write_text(f"{cls_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n", encoding="utf-8")


def _copy_negative(path: Path, split: str, index: int, dataset_root: Path) -> None:
    stem = _safe_stem(path.stem)
    name = f"hardneg_{index:07d}_{stem}{path.suffix.lower()}"
    dst_img = dataset_root / "images" / split / name
    dst_lbl = dataset_root / "labels" / split / f"{Path(name).stem}.txt"
    shutil.copy2(path, dst_img)
    dst_lbl.write_text("", encoding="utf-8")


def _write_data_yaml(dataset_root: Path) -> None:
    payload = {
        "path": str(dataset_root.resolve()),
        "train": "images/train",
        "val": "images/val",
        "names": {
            0: "customer_haircut",
            1: "staff_barber",
            2: "customer_wash",
        },
    }
    (dataset_root / "data.yaml").write_text(
        yaml.safe_dump(payload, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Prepare balanced salon YOLO dataset")
    p.add_argument("--haircut-dir", default="data/performance_feedback/haircut")
    p.add_argument("--staff-dir", default="data/staff_gallery/BARBER_UNIFORM")
    p.add_argument("--wash-dir", default="data/performance_feedback/customerwash")
    p.add_argument("--hard-negative-dir", action="append", default=["data/failed", "data/unknown_by_admin"])
    p.add_argument("--out", default="data/yolo_salon_dataset")
    p.add_argument("--val-ratio", type=float, default=0.15)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--autobox-model", default="yolov8n.pt")
    p.add_argument("--autobox-imgsz", type=int, default=640)
    p.add_argument("--device", default="auto")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    root = Path(__file__).resolve().parent.parent
    haircut_dir = (root / args.haircut_dir).resolve() if not Path(args.haircut_dir).is_absolute() else Path(args.haircut_dir)
    staff_dir = (root / args.staff_dir).resolve() if not Path(args.staff_dir).is_absolute() else Path(args.staff_dir)
    wash_dir = (root / args.wash_dir).resolve() if not Path(args.wash_dir).is_absolute() else Path(args.wash_dir)
    out_root = (root / args.out).resolve() if not Path(args.out).is_absolute() else Path(args.out)
    hard_neg_dirs = [(root / p).resolve() if not Path(p).is_absolute() else Path(p) for p in (args.hard_negative_dir or [])]

    samples_by_class = {
        "customer_haircut": _list_images(haircut_dir),
        "staff_barber": _list_images(staff_dir),
        "customer_wash": _list_images(wash_dir),
    }

    print("SOURCE_COUNTS")
    for cls_name, items in samples_by_class.items():
        print(f"  {cls_name}: {len(items)}")
    if all(len(v) == 0 for v in samples_by_class.values()):
        raise FileNotFoundError("No images found from haircut/staff/wash sources")

    balanced = _make_balanced(samples_by_class, seed=args.seed)
    for cls_name, arr in balanced.items():
        over = sum(1 for x in arr if x.is_oversampled)
        print(f"BALANCED {cls_name}: total={len(arr)} oversampled={over}")

    _prepare_root(out_root)

    device = _select_device(args.device)
    from ultralytics import YOLO

    autobox_model = YOLO(args.autobox_model)
    print(f"AUTOBOX model={args.autobox_model} device={device}")

    idx = 0
    for cls_name, arr in balanced.items():
        train_split, val_split = _split(arr, val_ratio=args.val_ratio, seed=args.seed + CLASS_TO_ID[cls_name])
        for sample in train_split:
            _copy_labeled_sample(sample, "train", idx, out_root, autobox_model, device, args.autobox_imgsz)
            idx += 1
        for sample in val_split:
            _copy_labeled_sample(sample, "val", idx, out_root, autobox_model, device, args.autobox_imgsz)
            idx += 1

    neg_images: List[Path] = []
    for d in hard_neg_dirs:
        neg_images.extend(_list_images(d))
    train_neg, val_neg = _split(
        [Sample(path=p, class_name="hard_negative", is_oversampled=False) for p in neg_images],
        val_ratio=args.val_ratio,
        seed=args.seed + 9001,
    )
    for item in train_neg:
        _copy_negative(item.path, "train", idx, out_root)
        idx += 1
    for item in val_neg:
        _copy_negative(item.path, "val", idx, out_root)
        idx += 1

    _write_data_yaml(out_root)
    print(f"DATASET_READY root={out_root}")
    print(f"HARD_NEGATIVE_ADDED train={len(train_neg)} val={len(val_neg)}")
    print(f"DATA_YAML {out_root / 'data.yaml'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
