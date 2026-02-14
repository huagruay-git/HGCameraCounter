"""
Build staff database with embeddings

Refactored to use centralized config
"""

import os
import sys
import json
import glob
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

import cv2
import numpy as np

# Optional heavy deps (import lazily)
try:
    import torch
except Exception:
    torch = None

try:
    from ultralytics import YOLO
except Exception:
    YOLO = None

# Add shared module to path
sys.path.insert(0, str(Path(__file__).parent))

from shared.config import Config
from shared.logger import setup_logger

# =========================
# SETUP
# =========================
CONFIG = Config("data/config/config.yaml")
logger = setup_logger("build_staff_db", CONFIG.get("paths", {}).get("logs", "logs"))

YOLO_CONFIG = CONFIG.get("yolo", {})
PATHS_CONFIG = CONFIG.get("paths", {})

YOLO_MODEL_PATH = os.path.join(PATHS_CONFIG.get("models", "models"), YOLO_CONFIG.get("model", "yolov8m.pt"))
STAFF_GALLERY_DIR = PATHS_CONFIG.get("staff_gallery", "data/staff_gallery")
STAFF_DB_PATH = PATHS_CONFIG.get("staff_db", "data/staff_gallery/staff_db.json")
CROPS_DIR = os.path.join(STAFF_GALLERY_DIR, "_crops")

IMG_EXTS = (".jpg", ".jpeg", ".png", ".webp", ".bmp")


# =========================
# UTILITIES
# =========================

def list_staff_images(staff_gallery_dir: str) -> Dict[str, List[str]]:
    """List all staff images grouped by staff ID"""
    staff_map: Dict[str, List[str]] = {}
    
    for staff_id in sorted(os.listdir(staff_gallery_dir)):
        staff_path = os.path.join(staff_gallery_dir, staff_id)
        
        if not os.path.isdir(staff_path) or staff_id.startswith("_"):
            continue
        
        imgs = []
        for ext in IMG_EXTS:
            imgs += glob.glob(os.path.join(staff_path, f"*{ext}"))
            imgs += glob.glob(os.path.join(staff_path, f"*{ext.upper()}"))
        
        imgs = sorted(list(set(imgs)))
        
        if imgs:
            staff_map[staff_id] = imgs
            logger.info(f"Found {len(imgs)} images for staff: {staff_id}")
    
    return staff_map


def ensure_dir(path: str):
    """Create directory if not exists"""
    os.makedirs(path, exist_ok=True)


def l2_normalize(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """L2 normalization"""
    n = np.linalg.norm(x) + eps
    return x / n


# =========================
# YOLO PERSON DETECTION
# =========================

def load_yolo_model():
    """Load YOLO model"""
    if YOLO is None:
        logger.error("ultralytics.YOLO is not installed")
        raise ImportError("ultralytics (YOLO) not available")

    if not os.path.exists(YOLO_MODEL_PATH):
        logger.error(f"YOLO model not found: {YOLO_MODEL_PATH}")
        raise FileNotFoundError(f"YOLO model: {YOLO_MODEL_PATH}")

    logger.info(f"Loading YOLO model: {YOLO_MODEL_PATH}")
    model = YOLO(YOLO_MODEL_PATH)
    return model


def detect_largest_person(yolo: YOLO, bgr: np.ndarray, conf: float = 0.35) -> Optional[Tuple[int, int, int, int, float]]:
    """Detect largest person bbox in image"""
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    results = yolo.predict(rgb, verbose=False, conf=conf, classes=[0])
    
    if not results or len(results) == 0:
        return None
    
    result = results[0]
    if result.boxes is None or len(result.boxes) == 0:
        return None
    
    best = None
    best_area = -1
    
    for box in result.boxes:
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        conf_score = float(box.conf[0]) if box.conf is not None else 0.0
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        
        area = max(0, x2 - x1) * max(0, y2 - y1)
        
        if area > best_area:
            best_area = area
            best = (x1, y1, x2, y2, conf_score)
    
    return best


def crop_person(bgr: np.ndarray, box_xyxy: Tuple[int, int, int, int], margin: float = 0.08) -> np.ndarray:
    """Crop person from image with margin"""
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


# =========================
# EMBEDDING EXTRACTOR
# =========================

class SimpleReIDExtractor:
    """Simple embedding extractor using timm backbone"""
    
    def __init__(self, model_name: str = "resnet50", device: str = "cpu"):
        try:
            import timm
            self.backbone = timm.create_model(
                model_name,
                pretrained=True,
                num_classes=0,
                global_pool="avg"
            )
            self.backbone = self.backbone.to(device)
            self.backbone.eval()
            self.device = device
            self.embed_dim = 2048 if "resnet50" in model_name else 512
            logger.info(f"Loaded timm model: {model_name}")
        except ImportError:
            logger.error("timm not installed, using random embeddings")
            self.backbone = None
            self.device = device
            self.embed_dim = 512
    
    def extract(self, bgr: np.ndarray) -> np.ndarray:
        """Extract embedding from image"""
        if self.backbone is None:
            return np.random.randn(self.embed_dim).astype(np.float32)
        
        # Preprocess
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        rgb = cv2.resize(rgb, (224, 224))
        
        # Normalize
        rgb_tensor = torch.from_numpy(rgb).float().permute(2, 0, 1) / 255.0
        rgb_tensor = rgb_tensor.unsqueeze(0).to(self.device)
        
        # Extract
        with torch.no_grad():
            embedding = self.backbone(rgb_tensor)
            embedding = embedding.squeeze(0).cpu().numpy()
        
        # Normalize
        embedding = l2_normalize(embedding.astype(np.float32))
        return embedding


# =========================
# STAFF DB BUILDER
# =========================

@dataclass
class BuildReport:
    """Report for staff DB building"""
    total_staff: int = 0
    total_images: int = 0
    success_count: int = 0
    failed_images: List[Dict[str, str]] = None
    staff_entries: List[Dict[str, Any]] = None
    
    def __post_init__(self):
        if self.failed_images is None:
            self.failed_images = []
        if self.staff_entries is None:
            self.staff_entries = []


def build_staff_db(
    staff_gallery_dir: str = STAFF_GALLERY_DIR,
    output_path: str = STAFF_DB_PATH,
    save_crops: bool = False
) -> BuildReport:
    """
    Build staff database with embeddings
    
    Args:
        staff_gallery_dir: path to staff gallery directory
        output_path: path to save staff_db.json
        save_crops: whether to save cropped images
    
    Returns:
        BuildReport
    """
    report = BuildReport()
    
    logger.info(f"Building staff database from: {staff_gallery_dir}")
    
    # Load YOLO
    try:
        yolo = load_yolo_model()
    except Exception as e:
        logger.error(f"Failed to load YOLO: {e}")
        report.failed_images.append({"reason": "YOLO model load failed"})
        return report
    
    # Load embedder
    if torch is not None:
        try:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            device = "cpu"
    else:
        device = "cpu"
    logger.info(f"Using device: {device}")
    embedder = SimpleReIDExtractor(device=device)
    
    # List staff images
    staff_map = list_staff_images(staff_gallery_dir)
    report.total_staff = len(staff_map)
    report.total_images = sum(len(imgs) for imgs in staff_map.values())
    
    if report.total_staff == 0:
        logger.warning("No staff found in gallery")
        return report
    
    logger.info(f"Found {report.total_staff} staff members with {report.total_images} images")
    
    # Create crops directory if needed
    if save_crops:
        ensure_dir(CROPS_DIR)
    
    # Process staff
    staff_db = {}
    
    for staff_id, image_paths in staff_map.items():
        logger.info(f"Processing {staff_id}...")
        
        embeddings = []
        
        for img_path in image_paths:
            try:
                bgr = cv2.imread(img_path)
                
                if bgr is None or bgr.size == 0:
                    logger.warning(f"Cannot read image: {img_path}")
                    report.failed_images.append({
                        "image": img_path,
                        "reason": "Cannot read image"
                    })
                    continue
                
                # Detect person
                box = detect_largest_person(yolo, bgr)
                
                if box is None:
                    logger.warning(f"No person detected: {img_path}")
                    report.failed_images.append({
                        "image": img_path,
                        "reason": "No person detected"
                    })
                    continue
                
                # Crop
                crop = crop_person(bgr, box[:4])
                
                # Save crop if requested
                if save_crops:
                    crop_path = os.path.join(CROPS_DIR, f"{staff_id}_{Path(img_path).stem}_crop.jpg")
                    cv2.imwrite(crop_path, crop)
                
                # Extract embedding
                embedding = embedder.extract(crop)
                embeddings.append(embedding.tolist())
                
                report.success_count += 1
            
            except Exception as e:
                logger.error(f"Error processing {img_path}: {e}")
                report.failed_images.append({
                    "image": img_path,
                    "reason": str(e)
                })
        
        if embeddings:
            staff_db[staff_id] = {
                "staff_id": staff_id,
                "num_images": len(embeddings),
                "embeddings": embeddings,
                "created_at": str(np.datetime64('now'))
            }
            report.staff_entries.append(staff_db[staff_id])
    
    # Save staff DB
    try:
        ensure_dir(os.path.dirname(output_path))
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(staff_db, f, indent=2)
        logger.info(f"Saved staff DB: {output_path}")
    except Exception as e:
        logger.error(f"Error saving staff DB: {e}")
    
    # Print report
    logger.info("=" * 50)
    logger.info(f"Staff DB Build Report")
    logger.info(f"Total staff: {report.total_staff}")
    logger.info(f"Total images: {report.total_images}")
    logger.info(f"Successful: {report.success_count}")
    logger.info(f"Failed: {len(report.failed_images)}")
    logger.info("=" * 50)
    
    if report.failed_images:
        logger.info("Failed images:")
        for failed in report.failed_images[:10]:  # Show first 10
            logger.info(f"  - {failed}")
    
    return report


# =========================
# MAIN
# =========================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Build staff database")
    parser.add_argument("--gallery", default=STAFF_GALLERY_DIR, help="Staff gallery directory")
    parser.add_argument("--output", default=STAFF_DB_PATH, help="Output staff_db.json path")
    parser.add_argument("--save-crops", action="store_true", help="Save cropped images")
    
    args = parser.parse_args()
    
    report = build_staff_db(
        staff_gallery_dir=args.gallery,
        output_path=args.output,
        save_crops=args.save_crops
    )
