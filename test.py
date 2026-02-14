import cv2
import threading
import time

# --- Configuration: เพิ่ม URL กล้องตามที่ต้องการ ---
CAMERAS = {
    "Camera 01": "rtsp://admin:112113114@192.168.1.24:554/ch01/0",
    "Camera 02": "rtsp://admin:112113114@192.168.1.83:554/ch01/0", # สมมติ IP ถัดไป
}

class CameraStream:
    def __init__(self, name, url):
        self.name = name
        self.url = url
        self.cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1) # ลด Buffer เพื่อลดดีเลย์
        self.frame = None
        self.ret = False
        self.running = True
        
        # เริ่มต้นเธรดเพื่อดึงภาพตลอดเวลา
        self.thread = threading.Thread(target=self.update, args=(), daemon=True)
        self.thread.start()

    def update(self):
        while self.running:
            if self.cap.isOpened():
                self.ret, self.frame = self.cap.read()
            else:
                print(f"⚠️ {self.name} connection lost. Retrying...")
                self.cap.open(self.url)
            time.sleep(0.01) # พักเล็กน้อยไม่ให้ CPU ทำงานหนักเกินไป

    def get_frame(self):
        return self.ret, self.frame

    def stop(self):
        self.running = False
        self.cap.release()

def main():
    streams = []
    
    # 1. เริ่มทำงานทุกกล้อง
    for name, url in CAMERAS.items():
        print(f"🚀 Starting {name}...")
        streams.append(CameraStream(name, url))

    print("✅ All cameras are running. Press 'q' to exit.")

    while True:
        for s in streams:
            success, frame = s.get_frame()
            
            if success and frame is not None:
                # ย่อขนาดหน้าต่างแสดงผลลงหน่อยถ้ามีหลายกล้อง
                display_frame = cv2.resize(frame, (640, 360))
                cv2.imshow(s.name, display_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # 2. ปิดระบบ
    for s in streams:
        s.stop()
    cv2.destroyAllWindows()
    print("🔌 All cameras stopped.")

if __name__ == "__main__":
    main()