#!/usr/bin/env python3
"""Build dataset from BARBER_UNIFORM/customer_by_admin, train YOLO, and apply model."""

from __future__ import annotations

import argparse
import random
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import yaml


PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from shared.config import Config


IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff", ".jp2"}


def _list_images_recursive(root: Path) -> List[Path]:
    if not root.exists() or not root.is_dir():
        return []
    out: List[Path] = []
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in IMG_EXTS:
            out.append(p)
    return sorted(out)


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


def _safe_stem(name: str) -> str:
    txt = "".join(ch if (ch.isalnum() or ch in ("-", "_")) else "_" for ch in name.strip())
    return txt.strip("_") or "img"


def _prepare_dataset_root(dataset_root: Path, append: bool) -> None:
    if dataset_root.exists() and (not append):
        shutil.rmtree(dataset_root, ignore_errors=True)
    (dataset_root / "images" / "train").mkdir(parents=True, exist_ok=True)
    (dataset_root / "images" / "val").mkdir(parents=True, exist_ok=True)
    (dataset_root / "labels" / "train").mkdir(parents=True, exist_ok=True)
    (dataset_root / "labels" / "val").mkdir(parents=True, exist_ok=True)


def _copy_split(
    files: Iterable[Path],
    split: str,
    cls_id: int,
    prefix: str,
    dataset_root: Path,
    start_index: int = 0,
) -> int:
    idx = int(start_index)
    for src in files:
        stem = _safe_stem(src.stem)
        dst_img = dataset_root / "images" / split / f"{prefix}_{idx:06d}_{stem}{src.suffix.lower()}"
        dst_lbl = dataset_root / "labels" / split / f"{dst_img.stem}.txt"
        shutil.copy2(src, dst_img)
        dst_lbl.write_text(f"{int(cls_id)} 0.5 0.5 1.0 1.0\n", encoding="utf-8")
        idx += 1
    return idx


def _write_dataset_yaml(dataset_root: Path) -> Path:
    data_yaml = dataset_root / "dataset.yaml"
    payload: Dict = {
        "path": str(dataset_root.resolve()),
        "train": "images/train",
        "val": "images/val",
        "names": {
            0: "person",
            1: "head_customer",
            2: "staff_uniform",
        },
    }
    data_yaml.write_text(yaml.safe_dump(payload, sort_keys=False, allow_unicode=True), encoding="utf-8")
    return data_yaml


def _resolve_model_path(project_root: Path, models_dir_cfg: str, model_name: str) -> Path:
    model_p = Path(str(model_name or "best.pt")).expanduser()
    if model_p.is_absolute():
        return model_p
    models_dir = Path(str(models_dir_cfg or "models"))
    if not models_dir.is_absolute():
        models_dir = project_root / models_dir
    return (models_dir / model_p.name).resolve()


def _backup_if_exists(path: Path) -> None:
    if not path.exists():
        return
    backup_dir = path.parent / "backups"
    backup_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = backup_dir / f"{path.stem}_before_role_retrain_{stamp}{path.suffix}"
    shutil.copy2(path, backup_path)
    print(f"BACKUP_MODEL {path} -> {backup_path}")


def _run_training(args: argparse.Namespace, data_yaml: Path) -> Path:
    train_script = PROJECT_ROOT / "scripts" / "train_yolo_head_staff.py"
    cmd = [
        sys.executable,
        str(train_script),
        "--data",
        str(data_yaml),
        "--model",
        str(args.model),
        "--epochs",
        str(int(args.epochs)),
        "--imgsz",
        str(int(args.imgsz)),
        "--batch",
        str(int(args.batch)),
        "--device",
        str(args.device),
        "--workers",
        str(int(args.workers)),
        "--patience",
        str(int(args.patience)),
        "--project",
        str(args.project),
        "--name",
        str(args.name),
    ]
    if args.exist_ok:
        cmd.append("--exist-ok")

    print(f"RUN_TRAIN {' '.join(cmd)}")
    proc = subprocess.Popen(
        cmd,
        cwd=str(PROJECT_ROOT),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    best_model: Path | None = None
    assert proc.stdout is not None
    for raw in proc.stdout:
        line = raw.rstrip("\r\n")
        print(line)
        if line.startswith("best_model="):
            best_model = Path(line.split("=", 1)[1].strip()).resolve()
    rc = proc.wait()
    if rc != 0:
        raise RuntimeError(f"Training failed (exit={rc})")

    if best_model is None or (not best_model.exists()):
        fallback = (PROJECT_ROOT / str(args.project) / str(args.name) / "weights" / "best.pt").resolve()
        if fallback.exists():
            best_model = fallback
    if best_model is None or (not best_model.exists()):
        raise FileNotFoundError("Training completed but best.pt was not found")
    return best_model


def parse_args(defaults: Dict) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Pipeline: build role dataset -> optional build role_db -> train YOLO -> apply model",
    )
    p.add_argument("--barber-dir", default=defaults["barber_dir"], help="BARBER_UNIFORM folder")
    p.add_argument("--customer-dir", default=defaults["customer_dir"], help="customer_by_admin folder")
    p.add_argument("--dataset-root", default=defaults["dataset_root"], help="Output YOLO dataset root")
    p.add_argument("--train-split", type=float, default=0.9, help="Train split ratio (0..1)")
    p.add_argument("--seed", type=int, default=42, help="Random seed for split")
    p.add_argument("--append", action="store_true", help="Append to existing dataset instead of recreate")

    p.add_argument("--model", default=defaults["base_model"], help="Base YOLO model for training")
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--device", default="auto", help="auto|cpu|mps|cuda")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--patience", type=int, default=20)
    p.add_argument("--project", default="runs/train")
    p.add_argument("--name", default="yolo_role_retrain")
    p.add_argument("--exist-ok", action="store_true")

    p.add_argument("--skip-build-role-db", action="store_true", help="Skip rebuilding role_db.json")
    p.add_argument("--skip-train", action="store_true", help="Only build dataset/role_db; no training")
    p.add_argument("--no-copy-runtime", action="store_true", help="Do not copy best.pt to runtime model path")
    p.add_argument(
        "--no-copy-worker-model",
        action="store_true",
        help="Do not copy best.pt to yolo.worker_model path (if different)",
    )
    return p.parse_args()


def main() -> int:
    cfg = Config(str(PROJECT_ROOT / "data" / "config" / "config.yaml"))
    cfg_paths = cfg.get("paths", {}) or {}
    yolo_cfg = cfg.get("yolo", {}) or {}

    defaults = {
        "barber_dir": str(Path(cfg_paths.get("staff_gallery", "data/staff_gallery")) / "BARBER_UNIFORM"),
        "customer_dir": str(Path(cfg_paths.get("customer_by_admin", "data/customer_by_admin"))),
        "dataset_root": str(PROJECT_ROOT / "data" / "yolo_role_dataset"),
        "base_model": str(yolo_cfg.get("discovery_model", "yolov8n.pt")),
    }
    args = parse_args(defaults)

    barber_dir = Path(args.barber_dir).expanduser()
    customer_dir = Path(args.customer_dir).expanduser()
    dataset_root = Path(args.dataset_root).expanduser()
    if not barber_dir.is_absolute():
        barber_dir = (PROJECT_ROOT / barber_dir).resolve()
    if not customer_dir.is_absolute():
        customer_dir = (PROJECT_ROOT / customer_dir).resolve()
    if not dataset_root.is_absolute():
        dataset_root = (PROJECT_ROOT / dataset_root).resolve()

    barber_images = _list_images_recursive(barber_dir)
    customer_images = _list_images_recursive(customer_dir)
    print(f"SOURCE barber={len(barber_images)} from {barber_dir}")
    print(f"SOURCE customer={len(customer_images)} from {customer_dir}")
    if not barber_images and not customer_images:
        raise FileNotFoundError("No images found in barber/customer folders")

    split_ratio = min(0.99, max(0.01, float(args.train_split)))
    rng = random.Random(int(args.seed))
    barber_train, barber_val = _split_train_val(barber_images, split_ratio, rng)
    customer_train, customer_val = _split_train_val(customer_images, split_ratio, rng)

    _prepare_dataset_root(dataset_root, append=bool(args.append))
    idx = 0
    idx = _copy_split(barber_train, "train", 2, "barber", dataset_root, start_index=idx)
    idx = _copy_split(barber_val, "val", 2, "barber", dataset_root, start_index=idx)
    idx = _copy_split(customer_train, "train", 0, "customer", dataset_root, start_index=idx)
    idx = _copy_split(customer_val, "val", 0, "customer", dataset_root, start_index=idx)
    data_yaml = _write_dataset_yaml(dataset_root)

    print(
        "DATASET_READY "
        f"root={dataset_root} train={(len(barber_train) + len(customer_train))} "
        f"val={(len(barber_val) + len(customer_val))} yaml={data_yaml}"
    )

    if not args.skip_build_role_db:
        try:
            from runtime.build_role_db import build_role_db

            role_db_path = Path(cfg_paths.get("role_db", "data/staff_gallery/barber_customer_db.json"))
            if not role_db_path.is_absolute():
                role_db_path = (PROJECT_ROOT / role_db_path).resolve()
            rep = build_role_db(
                barber_dir=str(barber_dir),
                customer_dir=str(customer_dir),
                output_path=str(role_db_path),
                save_crops=False,
            )
            print(
                f"ROLE_DB_DONE total={rep.total_images} success={rep.success_count} "
                f"barber={rep.role_counts.get('barber_uniform', 0)} "
                f"customer={rep.role_counts.get('customer', 0)} output={role_db_path}"
            )
        except Exception as e:
            print(f"ROLE_DB_ERROR {e}")
            raise

    if args.skip_train:
        print("SKIP_TRAIN requested")
        return 0

    best_model = _run_training(args, data_yaml)
    print(f"BEST_MODEL {best_model}")

    if args.no_copy_runtime:
        print("SKIP_COPY_RUNTIME requested")
        return 0

    models_dir_cfg = str(cfg_paths.get("models", "models"))
    runtime_model = _resolve_model_path(PROJECT_ROOT, models_dir_cfg, str(yolo_cfg.get("model", "best.pt")))
    _backup_if_exists(runtime_model)
    runtime_model.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(best_model, runtime_model)
    print(f"APPLY_MODEL runtime={runtime_model}")

    if not args.no_copy_worker_model:
        worker_name = str(yolo_cfg.get("worker_model", yolo_cfg.get("model", "best.pt")))
        worker_model = _resolve_model_path(PROJECT_ROOT, models_dir_cfg, worker_name)
        if worker_model.resolve() != runtime_model.resolve():
            _backup_if_exists(worker_model)
            worker_model.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(best_model, worker_model)
            print(f"APPLY_MODEL worker={worker_model}")

    print("PIPELINE_DONE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

