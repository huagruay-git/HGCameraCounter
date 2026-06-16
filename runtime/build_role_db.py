"""
Build role database (barber/customer) from admin-verified image folders.

This file creates a JSON knowledge base similar to staff_db.json, but with 2 roles:
- barber_uniform
- customer
"""

import json
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import numpy as np

try:
    from runtime.build_staff_db import (
        CONFIG,
        IMG_EXTS,
        PATHS_CONFIG,
        SimpleReIDExtractor,
        crop_person,
        detect_largest_person,
        ensure_dir,
        load_yolo_model,
        logger,
        l2_normalize,
        torch,
    )
except ModuleNotFoundError:
    # Allow running as: python runtime/build_role_db.py
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from runtime.build_staff_db import (  # type: ignore
        CONFIG,
        IMG_EXTS,
        PATHS_CONFIG,
        SimpleReIDExtractor,
        crop_person,
        detect_largest_person,
        ensure_dir,
        load_yolo_model,
        logger,
        l2_normalize,
        torch,
    )


@dataclass
class RoleBuildReport:
    total_images: int = 0
    success_count: int = 0
    failed_images: List[Dict[str, str]] = None
    role_counts: Dict[str, int] = None

    def __post_init__(self):
        if self.failed_images is None:
            self.failed_images = []
        if self.role_counts is None:
            self.role_counts = {"barber_uniform": 0, "customer": 0, "wash_customer": 0}


def _read_bgr_unicode_safe(path: Path) -> Optional[np.ndarray]:
    try:
        buf = np.fromfile(str(path), dtype=np.uint8)
        if buf.size == 0:
            return None
        return cv2.imdecode(buf, cv2.IMREAD_COLOR)
    except Exception:
        return None


def _list_images_recursive(root: Path) -> List[Path]:
    if not root.exists() or not root.is_dir():
        return []
    out: List[Path] = []
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in IMG_EXTS:
            out.append(p)
    return sorted(out)


def _encode_role_images(
    role_name: str,
    image_paths: List[Path],
    yolo_model,
    embedder: SimpleReIDExtractor,
    save_crops: bool = False,
    crops_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    role_images: List[str] = []
    embeddings: List[List[float]] = []

    for img_path in image_paths:
        bgr = _read_bgr_unicode_safe(img_path)
        if bgr is None or bgr.size == 0:
            logger.warning(f"[{role_name}] cannot read image: {img_path}")
            continue

        box = detect_largest_person(yolo_model, bgr)
        if box is None:
            crop = bgr.copy()
        else:
            crop = crop_person(bgr, box[:4])
            if crop is None or crop.size == 0:
                crop = bgr.copy()

        if save_crops and crops_dir is not None:
            try:
                ensure_dir(str(crops_dir))
                crop_path = crops_dir / f"{role_name}_{img_path.stem}_crop.jpg"
                cv2.imwrite(str(crop_path), crop)
            except Exception:
                pass

        emb = embedder.extract(crop)
        emb = l2_normalize(np.asarray(emb, dtype=np.float32))
        embeddings.append(emb.tolist())
        role_images.append(str(img_path))

    avg_vector: List[float] = []
    if embeddings:
        arr = np.asarray(embeddings, dtype=np.float32)
        avg_vector = l2_normalize(arr.mean(axis=0)).tolist()

    return {
        "role": role_name,
        "num_images": len(role_images),
        "images": role_images,
        "embeddings": embeddings,
        "avg_vector": avg_vector,
        "created_at": datetime.now().isoformat(),
    }


def build_role_db(
    barber_dir: str,
    customer_dir: str,
    output_path: str,
    wash_customer_dir: Optional[str] = None,
    save_crops: bool = False,
) -> RoleBuildReport:
    report = RoleBuildReport()
    barber_root = Path(barber_dir)
    customer_root = Path(customer_dir)
    wash_root = Path(wash_customer_dir) if wash_customer_dir else None
    out_path = Path(output_path)

    barber_images = _list_images_recursive(barber_root)
    customer_images = _list_images_recursive(customer_root)
    wash_images = _list_images_recursive(wash_root) if wash_root is not None else []
    report.total_images = len(barber_images) + len(customer_images) + len(wash_images)

    logger.info(
        f"Building role DB -> barber={len(barber_images)} customer={len(customer_images)} "
        f"wash_customer={len(wash_images)}"
    )

    if report.total_images == 0:
        logger.warning("No images found for role DB build")
        return report

    try:
        yolo = load_yolo_model()
    except Exception as e:
        report.failed_images.append({"image": "-", "reason": f"YOLO load failed: {e}"})
        logger.error(f"Cannot build role DB (YOLO): {e}")
        return report

    if torch is not None:
        try:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            device = "cpu"
    else:
        device = "cpu"
    embedder = SimpleReIDExtractor(device=device)

    crops_root = Path(PATHS_CONFIG.get("staff_gallery", "data/staff_gallery")) / "_role_crops"
    barber_entry = _encode_role_images(
        role_name="barber_uniform",
        image_paths=barber_images,
        yolo_model=yolo,
        embedder=embedder,
        save_crops=save_crops,
        crops_dir=(crops_root / "barber_uniform") if save_crops else None,
    )
    customer_entry = _encode_role_images(
        role_name="customer",
        image_paths=customer_images,
        yolo_model=yolo,
        embedder=embedder,
        save_crops=save_crops,
        crops_dir=(crops_root / "customer") if save_crops else None,
    )
    wash_customer_entry = None
    if wash_images:
        wash_customer_entry = _encode_role_images(
            role_name="wash_customer",
            image_paths=wash_images,
            yolo_model=yolo,
            embedder=embedder,
            save_crops=save_crops,
            crops_dir=(crops_root / "wash_customer") if save_crops else None,
        )

    report.role_counts["barber_uniform"] = int(barber_entry.get("num_images", 0))
    report.role_counts["customer"] = int(customer_entry.get("num_images", 0))
    if wash_customer_entry is not None:
        report.role_counts["wash_customer"] = int(wash_customer_entry.get("num_images", 0))
    report.success_count = int(sum(report.role_counts.values()))

    roles_payload = {
        "barber_uniform": barber_entry,
        "customer": customer_entry,
    }
    if wash_customer_entry is not None:
        roles_payload["wash_customer"] = wash_customer_entry

    role_db = {
        "version": "1.0",
        "created_at": datetime.now().isoformat(),
        "source": {
            "barber_dir": str(barber_root),
            "customer_dir": str(customer_root),
            "wash_customer_dir": str(wash_root) if wash_root is not None else "",
        },
        "roles": roles_payload,
    }

    try:
        ensure_dir(str(out_path.parent))
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(role_db, f, ensure_ascii=False, indent=2)
        logger.info(f"Saved role DB: {out_path}")
    except Exception as e:
        report.failed_images.append({"image": "-", "reason": f"Save failed: {e}"})
        logger.error(f"Failed writing role DB: {e}")

    return report


if __name__ == "__main__":
    import argparse

    default_staff_gallery = Path(PATHS_CONFIG.get("staff_gallery", "data/staff_gallery"))
    default_barber_dir = default_staff_gallery / "BARBER_UNIFORM"
    default_customer_dir = Path(PATHS_CONFIG.get("customer_by_admin", "data/customer_by_admin"))
    default_wash_customer_dir = Path(PATHS_CONFIG.get("customer_wash_by_admin", "data/customer_wash_by_admin"))
    default_output = Path(PATHS_CONFIG.get("role_db", "data/staff_gallery/barber_customer_db.json"))

    parser = argparse.ArgumentParser(description="Build barber/customer role DB JSON")
    parser.add_argument("--barber-dir", default=str(default_barber_dir), help="BARBER_UNIFORM image folder")
    parser.add_argument("--customer-dir", default=str(default_customer_dir), help="customer_by_admin image folder")
    parser.add_argument(
        "--wash-customer-dir",
        default=str(default_wash_customer_dir),
        help="customer_wash_by_admin image folder (optional)",
    )
    parser.add_argument("--output", default=str(default_output), help="Output role_db.json path")
    parser.add_argument("--save-crops", action="store_true", help="Save debug crops")
    args = parser.parse_args()

    rep = build_role_db(
        barber_dir=args.barber_dir,
        customer_dir=args.customer_dir,
        wash_customer_dir=args.wash_customer_dir,
        output_path=args.output,
        save_crops=args.save_crops,
    )
    logger.info(
        f"Role DB done: total={rep.total_images}, success={rep.success_count}, "
        f"barber={rep.role_counts.get('barber_uniform', 0)}, customer={rep.role_counts.get('customer', 0)}, "
        f"wash_customer={rep.role_counts.get('wash_customer', 0)}, "
        f"failed={len(rep.failed_images)}"
    )
