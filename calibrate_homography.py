# calibrate_homography.py
# Click 4 reference points on the FIRST frame of a video to build homography:
# - You click 4 points in this order:
#     1) top-left of "shop map"
#     2) top-right
#     3) bottom-right
#     4) bottom-left
# - Output: homography JSON mapping frame normalized (x,y) => shop_map normalized (0..1)
#
# Usage:
#   python calibrate_homography.py --video cam1.mp4 --out cam1_H.json
#   python calibrate_homography.py --video cam2.mp4 --out cam2_H.json

import argparse, json
import cv2
import numpy as np

PTS = []

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    cap = cv2.VideoCapture(args.video)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError("Cannot read first frame")

    h, w = frame.shape[:2]
    vis = frame.copy()

    def on_mouse(event, x, y, flags, param):
        nonlocal vis
        if event == cv2.EVENT_LBUTTONDOWN:
            nx, ny = x / w, y / h
            PTS.append([nx, ny])
            cv2.circle(vis, (x,y), 7, (0,255,0), -1)
            cv2.putText(vis, str(len(PTS)), (x+10, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
            cv2.imshow("calib", vis)
            if len(PTS) == 4:
                print("Got 4 points.")

    cv2.namedWindow("calib", cv2.WINDOW_NORMAL)
    cv2.setMouseCallback("calib", on_mouse)
    cv2.imshow("calib", vis)

    print("Click 4 points in order: TL, TR, BR, BL")
    print("Keys: r=reset, q=quit/save(if 4 points)")

    while True:
        key = cv2.waitKey(0) & 0xFF
        if key == ord("r"):
            PTS.clear()
            vis = frame.copy()
            cv2.imshow("calib", vis)
            print("reset")
        elif key == ord("q"):
            break

    cv2.destroyAllWindows()

    if len(PTS) != 4:
        print("Not enough points; nothing saved.")
        return

    src = np.array(PTS, dtype=np.float64)  # frame norm
    dst = np.array([
        [0.0, 0.0],  # TL
        [1.0, 0.0],  # TR
        [1.0, 1.0],  # BR
        [0.0, 1.0],  # BL
    ], dtype=np.float64)

    # homography mapping src -> dst
    H, _mask = cv2.findHomography(src, dst, method=0)
    if H is None:
        raise RuntimeError("findHomography failed")

    out = {"H": H.tolist(), "src_pts": PTS, "dst_pts": dst.tolist()}
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    print(f"Saved homography -> {args.out}")

if __name__ == "__main__":
    main()
