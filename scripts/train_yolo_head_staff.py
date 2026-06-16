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
    
    # Enhanced training features
    p.add_argument("--checkpoint-period", type=int, default=10, help="Save checkpoint every N epochs")
    p.add_argument("--min-delta", type=float, default=0.001, help="Minimum change in metrics to qualify as improvement")
    p.add_argument("--lr0", type=float, default=0.01, help="Initial learning rate")
    p.add_argument("--lrf", type=float, default=0.01, help="Final learning rate")
    p.add_argument("--momentum", type=float, default=0.937, help="SGD momentum")
    p.add_argument("--weight-decay", type=float, default=0.0005, help="Optimizer weight decay")
    p.add_argument("--warmup-epochs", type=int, default=3, help="Warmup epochs")
    p.add_argument("--warmup-momentum", type=float, default=0.8, help="Warmup momentum")
    p.add_argument("--warmup-bias-lr", type=float, default=0.1, help="Warmup bias learning rate")
    
    # Hyperparameter optimization mode
    p.add_argument("--hyperparameter-opt", action="store_true", help="Enable automatic hyperparameter optimization")
    p.add_argument("--opt-iters", type=int, default=50, help="Number of optimization iterations")
    
    return p.parse_args()


def main() -> None:
    args = parse_args()
    data_path = _resolve_dataset_yaml(args.data)
    if not data_path.exists():
        raise FileNotFoundError(f"dataset.yaml not found: {data_path}")

    try:
        cfg = yaml.safe_load(data_path.read_text(encoding="utf-8", errors="ignore")) or {}
    except Exception:
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
    import sys
    print(f"DATASET_CHECK train_dir={train_dir} images={train_count}", file=sys.stderr)
    print(f"DATASET_CHECK val_dir={val_dir} images={val_count}", file=sys.stderr)
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
        import sys
        print(f"DATASET_NOTE val split is empty; temporarily using train as val ({cfg[\"val\"]})", file=sys.stderr)
        import sys
        print(f"DATASET_NOTE val split is empty; temporarily using train as val ({cfg["val"]})", file=sys.stderr)

    device = _select_device(args.device)
    print(f"DEVICE_SELECTED: {device} (requested={args.device})")
    
    # Enhanced training configuration
    train_kwargs = {
        "data": train_data_arg,
        "epochs": args.epochs,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "device": device,
        "workers": args.workers,
        "patience": args.patience,
        "project": args.project,
        "name": args.name,
        "seed": args.seed,
        "exist_ok": args.exist_ok,
        "save_period": args.checkpoint_period,
        "min_delta": args.min_delta,
    }
    
    # Add optimizer parameters if they differ from defaults
    if args.lr0 != 0.01:
        train_kwargs["lr0"] = args.lr0
    if args.lrf != 0.01:
        train_kwargs["lrf"] = args.lrf
    if args.momentum != 0.937:
        train_kwargs["momentum"] = args.momentum
    if args.weight_decay != 0.0005:
        train_kwargs["weight_decay"] = args.weight_decay
    if args.warmup_epochs != 3:
        train_kwargs["warmup_epochs"] = args.warmup_epochs
    if args.warmup_momentum != 0.8:
        train_kwargs["warmup_momentum"] = args.warmup_momentum
    if args.warmup_bias_lr != 0.1:
        train_kwargs["warmup_bias_lr"] = args.warmup_bias_lr
    
    model = YOLO(args.model)
    
    if args.hyperparameter_opt:
        # Enable automatic hyperparameter optimization
        try:
            # Run hyperparameter evolution
            model.tune(
                data=train_data_arg,
                epochs=args.epochs,
                iterations=args.opt_iters,
                imgsz=args.imgsz,
                device=device,
                workers=args.workers,
                project=args.project,
                name=f"{args.name}_tune"
            )
            print("Hyperparameter optimization completed.")
            
            # After tuning, use the best parameters for final training
            import os
            tune_results_path = Path(args.project) / f"{args.name}_tune" / "tune_results.csv"
            if tune_results_path.exists():
                # Get the best hyperparameters from tuning results
                import pandas as pd
                df = pd.read_csv(tune_results_path)
                if not df.empty:
                    best_params = df.iloc[df['fitness'].idxmax()]
                    # Apply tuned hyperparameters
                    train_kwargs.update({
                        "lr0": best_params.get('lr0', args.lr0),
                        "lrf": best_params.get('lrf', args.lrf),
                        "momentum": best_params.get('momentum', args.momentum),
                        "weight_decay": best_params.get('weight_decay', args.weight_decay),
                        "warmup_epochs": int(best_params.get('warmup_epochs', args.warmup_epochs)),
                        "warmup_momentum": best_params.get('warmup_momentum', args.warmup_momentum),
                        "warmup_bias_lr": best_params.get('warmup_bias_lr', args.warmup_bias_lr),
                    })
        except Exception as e:
            print(f"Hyperparameter optimization failed: {e}")
            print("Proceeding with standard training...")
    
    try:
        results = model.train(**train_kwargs)
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
