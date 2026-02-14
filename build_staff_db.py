import os
import json
import glob
import argparse
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple

import cv2
import numpy as np

import torch
import torch.nn as nn

# YOLO from ultralytics
from ultralytics import YOLO


# ----------------------------
# Utils
# ----------------------------
IMG_EXTS = (".jpg", ".jpeg", ".png", ".webp", ".bmp")


def list_staff_images(staff_gallery_dir: str) -> Dict[str, List[str]]:
    staff_map: Dict[str, List[str]] = {}
    for staff_id in sorted(os.listdir(staff_gallery_dir)):
        staff_path = os.path.join(staff_gallery_dir, staff_id)
        if not os.path.isdir(staff_path):
            continue
        imgs = []
        for ext in IMG_EXTS:
            imgs += glob.glob(os.path.join(staff_path, f"*{ext}"))
            imgs += glob.glob(os.path.join(staff_path, f"*{ext.upper()}"))
        imgs = sorted(list(set(imgs)))
        if imgs:
            staff_map[staff_id] = imgs
    return staff_map


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def l2_normalize(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    n = np.linalg.norm(x) + eps
    return x / n


def iou(a, b) -> float:
    # a,b: (x1,y1,x2,y2)
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    iw = max(0, inter_x2 - inter_x1)
    ih = max(0, inter_y2 - inter_y1)
    inter = iw * ih
    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = area_a + area_b - inter + 1e-6
    return inter / union


# ----------------------------
# Person crop using YOLO
# ----------------------------
def detect_largest_person_box(yolo: YOLO, bgr: np.ndarray, conf: float = 0.35) -> Optional[Tuple[int,int,int,int,float]]:
    """
    Return largest person bbox in xyxy + score
    """
    # ultralytics expects RGB
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    res = yolo.predict(rgb, verbose=False, conf=conf, classes=[0])  # class 0 = person
    if not res or len(res) == 0:
        return None
    r0 = res[0]
    if r0.boxes is None or len(r0.boxes) == 0:
        return None

    best = None
    best_area = -1
    for box in r0.boxes:
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        score = float(box.conf[0].item()) if box.conf is not None else 0.0
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        area = max(0, x2 - x1) * max(0, y2 - y1)
        if area > best_area:
            best_area = area
            best = (x1, y1, x2, y2, score)
    return best


def crop_with_margin(bgr: np.ndarray, box_xyxy, margin: float = 0.08) -> np.ndarray:
    h, w = bgr.shape[:2]
    x1, y1, x2, y2 = box_xyxy
    bw = x2 - x1
    bh = y2 - y1
    mx = int(bw * margin)
    my = int(bh * margin)
    x1 = max(0, x1 - mx)
    y1 = max(0, y1 - my)
    x2 = min(w - 1, x2 + mx)
    y2 = min(h - 1, y2 + my)
    return bgr[y1:y2, x1:x2].copy()


# ----------------------------
# Embedding backends
# ----------------------------
class TimmReID(nn.Module):
    """
    Fallback re-id style embedding using timm backbone
    """
    def __init__(self, model_name="resnet50", embed_dim=2048):
        super().__init__()
        import timm
        self.backbone = timm.create_model(model_name, pretrained=True, num_classes=0, global_pool="")
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        # infer dim
        with torch.no_grad():
            x = torch.zeros(1, 3, 256, 128)
            y = self.backbone(x)
            if y.dim() == 4:
                y = self.pool(y).flatten(1)
            dim = y.shape[1]
        self.embed_dim = dim

    def forward(self, x):
        y = self.backbone(x)
        if y.dim() == 4:
            y = self.pool(y).flatten(1)
        return y


def load_embedder(device: str = "cpu"):
    """
    Prefer torchreid if installed; else fallback to timm
    """
    # 1) try torchreid
    try:
        import torchreid
        from torchreid.utils import FeatureExtractor
        extractor = FeatureExtractor(
            model_name="osnet_x1_0",
            model_path="",
            device=device
        )
        return ("torchreid", extractor, 512)
    except Exception:
        pass

    # 2) fallback timm
    model = TimmReID(model_name="resnet50")
    model.eval().to(device)
    return ("timm", model, model.embed_dim)


def preprocess_person_crop(bgr_crop: np.ndarray, out_hw=(256, 128)) -> torch.Tensor:
    """
    Standard ReID crop: H=256, W=128
    """
    H, W = out_hw
    rgb = cv2.cvtColor(bgr_crop, cv2.COLOR_BGR2RGB)
    rgb = cv2.resize(rgb, (W, H), interpolation=cv2.INTER_LINEAR)
    x = rgb.astype(np.float32) / 255.0
    # normalize ImageNet-ish
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    x = (x - mean) / std
    x = np.transpose(x, (2, 0, 1))  # CHW
    return torch.from_numpy(x).unsqueeze(0)  # 1x3xHxW


def embed_one(backend: str, embedder, device: str, bgr_crop: np.ndarray) -> np.ndarray:
    x = preprocess_person_crop(bgr_crop).to(device)
    with torch.no_grad():
        if backend == "torchreid":
            feat = embedder(bgr_crop)  # FeatureExtractor accepts np image or path
            if isinstance(feat, torch.Tensor):
                feat = feat.detach().cpu().numpy()
            feat = np.asarray(feat).reshape(-1)
        else:
            feat = embedder(x).detach().cpu().numpy().reshape(-1)

    feat = l2_normalize(feat.astype(np.float32))
    return feat


# ----------------------------
# Main builder
# ----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gallery", default="staff_gallery", help="staff_gallery dir")
    ap.add_argument("--out", default="staff_gallery/staff_db.json", help="output staff_db.json path")
    ap.add_argument("--yolo", default="yolov8n.pt", help="YOLO model path/name")
    ap.add_argument("--device", default="cpu", help="cpu | mps | cuda")
    ap.add_argument("--conf", type=float, default=0.35, help="YOLO person confidence")
    ap.add_argument("--min_images", type=int, default=2, help="min images per staff to include")
    ap.add_argument("--save_crops", action="store_true", help="save person crops for debug")
    ap.add_argument("--crops_dir", default="staff_gallery/_crops", help="debug crops dir")
    args = ap.parse_args()

    if not os.path.isdir(args.gallery):
        raise SystemExit(f"Gallery not found: {args.gallery}")

    print("🧠 Building staff_db.json ...")
    print(f"📁 gallery={args.gallery}")
    print(f"🤖 yolo={args.yolo} conf={args.conf}")
    print(f"🖥️ device={args.device}")

    # load YOLO
    yolo = YOLO(args.yolo)

    # load embedder
    backend, embedder, dim = load_embedder(args.device)
    print(f"✅ Embedding backend={backend} dim={dim}")

    staff_map = list_staff_images(args.gallery)
    if not staff_map:
        raise SystemExit("No staff folders/images found.")

    items: List[Dict[str, Any]] = []

    if args.save_crops:
        ensure_dir(args.crops_dir)

    total_vec = 0
    for staff_id, img_paths in staff_map.items():
        print(f"\n👤 {staff_id} images={len(img_paths)}")
        if len(img_paths) < args.min_images:
            print(f"  ⚠️ skip (need >= {args.min_images})")
            continue

        vectors = []
        used_imgs = 0

        for p in img_paths:
            bgr = cv2.imread(p)
            if bgr is None:
                print(f"  ⚠️ read fail: {p}")
                continue

            det = detect_largest_person_box(yolo, bgr, conf=args.conf)
            if det is None:
                print(f"  ⚠️ no person: {os.path.basename(p)}")
                continue

            x1, y1, x2, y2, score = det
            if (x2 - x1) < 40 or (y2 - y1) < 60:
                print(f"  ⚠️ bbox too small: {os.path.basename(p)}")
                continue

            crop = crop_with_margin(bgr, (x1, y1, x2, y2), margin=0.08)

            if args.save_crops:
                outp = os.path.join(args.crops_dir, f"{staff_id}__{os.path.basename(p)}")
                cv2.imwrite(outp, crop)

            try:
                vec = embed_one(backend, embedder, args.device, crop)
                vectors.append(vec.tolist())
                used_imgs += 1
            except Exception as e:
                print(f"  ❌ embed fail: {os.path.basename(p)} err={e}")

        if used_imgs == 0:
            print("  ❌ no usable images")
            continue

        # average vector (still keep all vectors too)
        avg = l2_normalize(np.mean(np.array(vectors, dtype=np.float32), axis=0)).tolist()

        items.append({
            "staff_id": staff_id,
            "num_images": len(img_paths),
            "num_vectors": len(vectors),
            "avg_vector": avg,
            "vectors": vectors,
            "image_paths": [os.path.relpath(x, start=os.path.dirname(args.out)) for x in img_paths],
        })

        total_vec += len(vectors)
        print(f"  ✅ vectors={len(vectors)} (used_images={used_imgs})")

    db = {
        "version": "1.0",
        "backend": backend,
        "embed_dim": dim,
        "items": items
    }

    ensure_dir(os.path.dirname(args.out) or ".")
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(db, f, ensure_ascii=False, indent=2)

    print("\n📦 그래็บผลลัพธ์")
    print(f"✅ wrote: {args.out}")
    print(f"👥 staff={len(items)} | vectors_total={total_vec}")

    if len(items) == 0:
        print("⚠️ staff_db.json ยังว่างอยู่: ตรวจรูป/โฟลเดอร์/YOLO detect person ไม่เจอ")
    else:
        print("➡️ ต่อไปให้รัน edge_agent.py ใหม่ แล้วเช็ค log ต้องไม่ใช่ 0 embeddings แล้ว")


if __name__ == "__main__":
    main()
