import cv2
import json
import os
import math
import numpy as np
from datetime import datetime

# --- Configuration: รายชื่อกล้อง ---
CAMERAS = {
    "Camera_01": "rtsp://admin:112113114@192.168.1.24:554/ch01/0",
    "Camera_02": "rtsp://admin:112113114@192.168.1.83:554/ch01/0",
    "Camera_03": "rtsp://admin:112113114@192.168.1.91:554/ch01/0",
}

# ----------------------------
# Utils
# ----------------------------
def order_points_clockwise(pts):
    if len(pts) != 4: return pts
    cx = sum(p[0] for p in pts) / 4.0
    cy = sum(p[1] for p in pts) / 4.0
    def ang(p): return math.atan2(p[1] - cy, p[0] - cx)
    pts_sorted = sorted(pts, key=ang)
    idx = min(range(4), key=lambda i: pts_sorted[i][0] + pts_sorted[i][1])
    return pts_sorted[idx:] + pts_sorted[:idx]

def to_norm_points(pts_px, w, h, round_n=4):
    return [{"x": round(x / w, round_n), "y": round(y / h, round_n)} for x, y in pts_px]

def draw_polygon(img, pts_px, color, thickness=3, fill_alpha=0.0):
    if len(pts_px) < 2: return img
    out = img.copy()
    poly = np.array(pts_px, dtype=np.int32).reshape((-1, 1, 2))
    if fill_alpha > 0:
        overlay = out.copy()
        cv2.fillPoly(overlay, [poly], color)
        out = cv2.addWeighted(overlay, fill_alpha, out, 1 - fill_alpha, 0)
    cv2.polylines(out, [poly], isClosed=True, color=color, thickness=thickness)
    return out

def draw_label(img, pts_px, label, color):
    out = img.copy()
    cx = int(sum(p[0] for p in pts_px) / len(pts_px))
    cy = int(sum(p[1] for p in pts_px) / len(pts_px))
    cv2.putText(out, label, (cx - 10, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
    return out

# ----------------------------
# Camera Selection
# ----------------------------
print("\n--- เลือกกล้องที่ต้องการตั้งค่าโซน ---")
cam_keys = list(CAMERAS.keys())
for i, name in enumerate(cam_keys):
    print(f"{i}: {name} ({CAMERAS[name]})")

while True:
    try:
        choice = int(input("\nใส่หมายเลขกล้อง: "))
        if 0 <= choice < len(cam_keys):
            SELECTED_CAM_NAME = cam_keys[choice]
            RTSP_URL = CAMERAS[SELECTED_CAM_NAME]
            break
    except ValueError: pass
    print("หมายเลขไม่ถูกต้อง กรุณาเลือกใหม่")

OUT_JSON = f"zones_{SELECTED_CAM_NAME}.json"

# ----------------------------
# Capture Frame from RTSP
# ----------------------------
print(f"\nกำลังเชื่อมต่อกับ {SELECTED_CAM_NAME}...")
cap = cv2.VideoCapture(RTSP_URL)
# เคลียร์ buffer เพื่อให้ได้ภาพล่าสุด
for _ in range(5): cap.grab() 
ok, frame = cap.read()
cap.release()

if not ok:
    print(f"❌ ไม่สามารถดึงภาพจากกล้องได้: {RTSP_URL}")
    exit()

h, w = frame.shape[:2]
base = frame.copy()
show = frame.copy()

ZONES = []       
CUR_POINTS = []  

HELP = f"""
กล้องที่เลือก: {SELECTED_CAM_NAME}
คลิก 4 จุด -> วาดโซน
คีย์ลัด:
  n = ตั้งชื่อ + บันทึกโซน (เช่น WAIT, WASH, CHAIR_1, SHOP, STAFF_AREA)
  u = undo
  r = reset จุด
  s = save ลงไฟล์ {OUT_JSON}
  q = ออก
"""

def redraw():
    global show
    show = base.copy()
    for z in ZONES:
        pts = z["points_px"]
        show = draw_polygon(show, pts, color=(0, 0, 255), thickness=2)
        show = draw_label(show, pts, z["name"], color=(0, 0, 255))
    for i, (x, y) in enumerate(CUR_POINTS, start=1):
        cv2.circle(show, (x, y), 5, (0, 255, 0), -1)
    if len(CUR_POINTS) == 4:
        pts = order_points_clockwise(CUR_POINTS)
        show = draw_polygon(show, pts, color=(0, 255, 0), thickness=2, fill_alpha=0.2)
    cv2.imshow("Zone Picker", show)

def on_mouse(event, x, y, flags, param):
    global CUR_POINTS
    if event == cv2.EVENT_LBUTTONDOWN:
        if len(CUR_POINTS) < 4:
            CUR_POINTS.append((x, y))
            redraw()

print(HELP)
cv2.namedWindow("Zone Picker", cv2.WINDOW_NORMAL)
cv2.setMouseCallback("Zone Picker", on_mouse)
redraw()

while True:
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"): break
    if key == ord("u") and CUR_POINTS:
        CUR_POINTS.pop()
        redraw()
    if key == ord("r"):
        CUR_POINTS.clear()
        redraw()
    if key == ord("n") and len(CUR_POINTS) == 4:
        name = input(f"\nชื่อโซนสำหรับ {SELECTED_CAM_NAME}: ").strip().upper()
        if name:
            pts = order_points_clockwise(CUR_POINTS)
            poly = to_norm_points(pts, w, h)
            ZONES.append({
                "name": name,
                "polygon_json": poly,
                "points_px": pts
            })
            CUR_POINTS.clear()
            redraw()
            print(f"บันทึกชั่วคราว: {name}")
    if key == ord("s"):
        out_data = [{"name": z["name"], "polygon_json": z["polygon_json"]} for z in ZONES]
        with open(OUT_JSON, "w", encoding="utf-8") as f:
            json.dump(out_data, f, ensure_ascii=False, indent=2)
        print(f"\n✅ บันทึกสำเร็จ! ไฟล์: {OUT_JSON}")

cv2.destroyAllWindows()