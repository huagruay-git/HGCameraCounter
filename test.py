import cv2
import threading
import time
from collections import deque
import numpy as np

# --- Configuration: เพิ่ม URL กล้องตามที่ต้องการ ---
CAMERAS = {
    "Camera 01": "rtsp://admin:112113114@192.168.1.24:554/ch01/0",
    "Camera 02": "rtsp://admin:112113114@192.168.1.83:554/ch01/0", # สมมติ IP ถัดไป
}

class CameraStream:
    def __init__(self, name, url):
        self.name = name
        self.url = url
        self.cap = None
        self.frame = None
        self.ret = False
        self.running = True
        self.status = "INITIALIZING"
        self.last_success_ts = 0
        self.fps_tracker = deque(maxlen=30) # Store timestamps of last 30 frames
        
        # เริ่มต้นเธรดเพื่อดึงภาพตลอดเวลา
        self.thread = threading.Thread(target=self.update, args=(), daemon=True)
        self.thread.start()

    def _connect(self):
        self.status = f"CONNECTING to {self.url}"
        print(f"[{time.strftime('%H:%M:%S')}] {self.name}: {self.status}")
        self.cap = cv2.VideoCapture(self.url, cv2.CAP_FFMPEG)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        if self.cap.isOpened():
            self.status = "CONNECTED"
            self.last_success_ts = time.time()
            print(f"[{time.strftime('%H:%M:%S')}] {self.name}: {self.status}")
        else:
            self.status = "CONNECTION FAILED"
            print(f"[{time.strftime('%H:%M:%S')}] {self.name}: {self.status}")
            if self.cap:
                self.cap.release()
            self.cap = None

    def update(self):
        self._connect()
        while self.running:
            if self.cap and self.cap.isOpened():
                self.ret, self.frame = self.cap.read()
                if self.ret:
                    self.last_success_ts = time.time()
                    self.fps_tracker.append(self.last_success_ts)
                else:
                    # Frame read failed, maybe stream ended or temporary glitch
                    if time.time() - self.last_success_ts > 5.0: # If no frame for 5s
                        print(f"[{time.strftime('%H:%M:%S')}] ⚠️ {self.name} frame read failed. Reconnecting...")
                        if self.cap:
                            self.cap.release()
                        self.cap = None
                        time.sleep(2) # Wait before reconnecting
                        self._connect()
            else:
                # Not connected, try to connect
                time.sleep(5)
                self._connect()

            time.sleep(0.005) # Small sleep to yield CPU

    def get_frame(self):
        return self.ret, self.frame

    def get_fps(self):
        if len(self.fps_tracker) < 2:
            return 0.0
        return (len(self.fps_tracker) - 1) / (self.fps_tracker[-1] - self.fps_tracker[0] + 1e-6)

    def stop(self):
        self.running = False
        if self.cap:
            self.cap.release()
        self.cap = None

def main():
    streams = []
    
    # 1. เริ่มทำงานทุกกล้อง
    for name, url in CAMERAS.items():
        if not url: continue
        print(f"🚀 Starting {name}...")
        streams.append(CameraStream(name, url))

    print("\n✅ All cameras are running. Press 'q' to exit.")
    print("This script tests camera stream stability without heavy processing.")
    print("If this runs smoothly but the main app fails, the problem is likely CPU/memory overload.\n")


    while True:
        all_frames = []
        for s in streams:
            success, frame = s.get_frame()
            
            if success and frame is not None:
                display_frame = frame.copy()
                # Add status text
                fps = s.get_fps()
                status_text = f"[{s.name}] FPS: {fps:.1f} | Status: {s.status}"
                cv2.putText(display_frame, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
                all_frames.append(display_frame)
            else:
                # Create a black frame with error message
                error_frame = np.zeros((360, 640, 3), dtype=np.uint8)
                status_text = f"[{s.name}] | Status: {s.status}"
                cv2.putText(error_frame, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA)
                all_frames.append(error_frame)

        # Combine frames into a single window if possible
        if len(all_frames) > 0:
            # Resize all to a standard size for stacking
            h, w = 360, 640
            resized_frames = [cv2.resize(f, (w, h)) for f in all_frames]
            
            montage = np.hstack(resized_frames) if len(resized_frames) > 1 else resized_frames[0]

            cv2.imshow("Camera Stability Test", montage)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # 2. ปิดระบบ
    for s in streams:
        s.stop()
    cv2.destroyAllWindows()
    print("🔌 All cameras stopped.")

if __name__ == "__main__":
    main()