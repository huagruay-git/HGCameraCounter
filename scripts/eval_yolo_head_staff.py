#!/usr/bin/env python3
"""Evaluate Ultralytics YOLO model and output percentage metrics for GUI."""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

import yaml

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff", ".jp2"}


def _count_images(dir_path: Path) -> int:
    if not dir_path.exists():
        return 0
    return sum(1 for p in dir_path.rglob("*") if p.is_file() and p.suffix.lower() in IMG_EXTS)


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


def resolve_weights_path(raw_path: str) -> Path:
    p = Path(raw_path).expanduser()
    if p.is_absolute() and p.exists():
        return p
    # Try cwd-relative first
    if p.exists():
        return p.resolve()

    # Try project-root relative (script is in HGCameraCounter/scripts)
    project_root = Path(__file__).resolve().parent.parent
    proj_candidate = project_root / raw_path
    if proj_candidate.exists():
        return proj_candidate.resolve()

    # Fallback: newest best.pt under common Ultralytics output roots.
    search_roots = [
        project_root / "runs" / "train",
        project_root / "runs" / "detect",
        Path.home() / "runs" / "train",
        Path.home() / "runs" / "detect",
    ]
    found: list[Path] = []
    for root in search_roots:
        if root.exists():
            found.extend(root.glob("**/weights/best.pt"))
    candidates = sorted(found, key=lambda x: x.stat().st_mtime, reverse=True)
    if candidates:
        return candidates[0].resolve()
    return p


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate YOLO model")
    p.add_argument("--weights", required=True, help="Path to trained weights (.pt)")
    p.add_argument("--data", required=True, help="Path to dataset.yaml")
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--device", default="auto", help="auto|cpu|mps|cuda")
    p.add_argument("--split", default="val", choices=["train", "val", "test"])
    p.add_argument("--verbose", action="store_true", help="Verbose output with detailed metrics")
    p.add_argument("--per-class", action="store_true", help="Include per-class metrics in output")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    try:
        from ultralytics import YOLO
    except Exception as e:
        raise ImportError("ultralytics is required. Install with: pip install ultralytics") from e

    weights = resolve_weights_path(args.weights)
    data_yaml = _resolve_dataset_yaml(args.data)
    if not weights.exists():
        raise FileNotFoundError(
            f"weights not found: {weights}\n"
            "Tip: train first or use Browse to select your .pt weights file."
        )
    if not data_yaml.exists():
        raise FileNotFoundError(f"dataset yaml not found: {data_yaml}")
    device = _select_device(args.device)
    print(f"DEVICE_SELECTED: {device} (requested={args.device})")

    cfg = yaml.safe_load(data_yaml.read_text(encoding="utf-8")) or {}
    base = data_yaml.parent.resolve()
    project_root = Path(__file__).resolve().parent.parent
    path_cfg = Path(cfg.get("path", "."))
    if path_cfg.is_absolute():
        path_root = path_cfg
    else:
        candidates = [(base / path_cfg).resolve(), (project_root / path_cfg).resolve(), base]
        path_root = next((c for c in candidates if c.exists()), candidates[0])

    train_rel = str(cfg.get("train", "images/train"))
    val_rel = str(cfg.get("val", "images/val"))
    train_count = _count_images((path_root / train_rel).resolve())
    val_count = _count_images((path_root / val_rel).resolve())
    print(f"DATASET_CHECK train_images={train_count} val_images={val_count}")
    if train_count <= 0:
        raise FileNotFoundError("No images found in train split.")

    eval_data_arg = str(data_yaml)
    split = args.split
    tmp_yaml_path: Path | None = None
    if split == "val" and val_count <= 0:
        cfg["val"] = cfg.get("train", "images/train")
        fd, tmp_name = tempfile.mkstemp(prefix="dataset_eval_bootstrap_", suffix=".yaml")
        Path(tmp_name).write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
        tmp_yaml_path = Path(tmp_name)
        eval_data_arg = str(tmp_yaml_path)
        print(f"DATASET_NOTE val split is empty; using train as val ({cfg['val']})")

    model = YOLO(str(weights))
    try:
        results = model.val(
            data=eval_data_arg,
            split=split,
            imgsz=args.imgsz,
            batch=args.batch,
            device=device,
            verbose=args.verbose,
        )
    finally:
        if tmp_yaml_path is not None:
            try:
                tmp_yaml_path.unlink(missing_ok=True)
            except Exception:
                pass

    precision = float(getattr(results.box, "mp", 0.0))
    recall = float(getattr(results.box, "mr", 0.0))
    map50 = float(getattr(results.box, "map50", 0.0))
    map50_95 = float(getattr(results.box, "map", 0.0))

    metrics = {
        "precision": precision,
        "recall": recall,
        "map50": map50,
        "map50_95": map50_95,
        "precision_pct": precision * 100.0,
        "recall_pct": recall * 100.0,
        "map50_pct": map50 * 100.0,
        "map50_95_pct": map50_95 * 100.0,
    }
    
    # Add per-class metrics if requested
    if args.per_class and hasattr(results.box, 'ap_class_index'):
        try:
            # Get class-specific metrics
            class_metrics = {}
            if hasattr(results.box, 'ap_class_index') and hasattr(results.box, 'ap50'):
                for i, class_index in enumerate(results.box.ap_class_index):
                    class_name = cfg.get('names', {}).get(class_index, f'class_{class_index}')
                    class_metrics[class_name] = {
                        'ap50': float(results.box.ap50[i]),
                        'ap50_95': float(results.box.ap[i]),
                        'precision': float(results.box.p[i]),
                        'recall': float(results.box.r[i]),
                    }
            metrics['per_class'] = class_metrics
        except Exception as e:
            print(f"Warning: Could not extract per-class metrics: {e}")
    
    print(f"USING_WEIGHTS: {weights}")
    print("METRICS_JSON:", json.dumps(metrics, ensure_ascii=False))
    
    # Print detailed metrics if verbose
    if args.verbose:
        print("\nDetailed Metrics:")
        print(f"  Precision: {metrics['precision_pct']:.2f}%")
        print(f"  Recall: {metrics['recall_pct']:.2f}%")
        print(f"  mAP@0.5: {metrics['map50_pct']:.2f}%")
        print(f"  mAP@0.5:0.95: {metrics['map50_95_pct']:.2f}%")
        
        if args.per_class and 'per_class' in metrics:
            print("\nPer-Class Metrics:")
            for class_name, class_data in metrics['per_class'].items():
                print(f"  {class_name}:")
                print(f"    AP@0.5: {class_data['ap50']*100:.2f}%")
                print(f"    AP@0.5:0.95: {class_data['ap50_95']*100:.2f}%")
                print(f"    Precision: {class_data['precision']*100:.2f}%")
                print(f"    Recall: {class_data['recall']*100:.2f}%")


if __name__ == "__main__":
    main()
