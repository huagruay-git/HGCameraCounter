Requirement ของโปรเจกต์: “HG Camera Counter Setup Suite”
1) วัตถุประสงค์

สร้างชุดโปรแกรมสำหรับ “ติดตั้ง/ตั้งค่า/ตรวจสอบความพร้อม/รันระบบนับลูกค้า” จากกล้อง RTSP หลายตัวในแต่ละสาขา พร้อมเครื่องมือประกอบ (วาดโซน, สร้าง staff_db, ตรวจ Supabase, ดูสถานะออนไลน์)

2) ขอบเขตระบบ (Scope)

2.1 ส่วนประกอบหลัก

Controller App (GUI)

ใช้ตั้งค่า/ทดสอบ/จัดการกล้อง/วาดโซน/สร้าง staff_db/ดูสถานะ/สั่งเริ่มหยุดระบบ

Runtime Service

ตัวรันตรวจนับจริง (YOLO + logic zone + dwell time + logging + ส่ง Supabase)

ทำงานแบบ background / auto-start / watchdog

Installer / Setup Package

ติดตั้งแบบคลิกเดียว พร้อม dependency และสร้าง service/shortcut

2.2 ขอบเขตข้อมูล

config กลาง (branch, cameras, thresholds, paths, model)

zones per camera

staff_gallery และ staff_db

logs / reports / snapshots

3) Functional Requirements (FR)
FR-01: Setup Wizard

ผู้ใช้สามารถทำตามขั้นตอน (Wizard) เพื่อ setup ระบบได้

Step ขั้นต่ำ:

ตั้งค่า Supabase + ทดสอบ

เพิ่ม/ลบกล้อง RTSP + ทดสอบสตรีม

วาดโซน + บันทึก zones.json

สร้าง/จัดการ staff_gallery + build staff_db.json

ตรวจความพร้อมระบบ (Health checks)

ติดตั้ง/เริ่มระบบ runtime + ตั้ง auto-start

FR-02: Camera Management (เพิ่ม/ลดกล้อง)

เพิ่มกล้องได้ไม่จำกัด (ตามสเปคเครื่อง)

แต่ละกล้องมี: name, rtsp_url, enabled, zones_file

ทดสอบ RTSP: connect ได้ไหม, ได้ frame ไหม, fps/latency, snapshot preview

FR-03: Zone Editor

เลือกกล้อง → แสดงภาพล่าสุด (snapshot หรือ live frame)

วาด polygon ได้หลายโซน เช่น:

CHAIR_1..N

WAIT

WASH

STAFF_AREA (optional)

Save/Load zones JSON

มี “validation” เช่น polygon ไม่ทับซ้อน/ไม่ว่าง/จุดขั้นต่ำ 3 จุด

FR-04: Build Staff DB (สร้าง staff_db.json)

เลือกโฟลเดอร์ staff_gallery

ตรวจคุณภาพรูปขั้นต่ำ (จำนวน/ขนาด/เบลอมากเกิน)

กด build → สร้าง staff_db.json

แสดงผลลัพธ์: staff กี่คน, embeddings กี่รายการ, รูปที่ตกหล่นเพราะอะไร

(ทางเลือก) Save crops สำหรับตรวจสอบ

FR-05: System Health Checks

ต้องมีปุ่ม “Run Diagnostics” แล้วรายงานสถานะ:

Internet/DNS/Ping

RTSP per camera (OK/Fail + reason)

Model file ready

Permission/Path (write snapshots/reports)

GPU device (cpu/mps/cuda) และ fps estimate

Supabase connect + identify branch ได้จริง

FR-06: Supabase Integration

ทดสอบ connection (อ่าน/เขียนที่จำเป็น)

แสดง “เชื่อมต่อได้” และ “สาขาไหน (branch_code/name)”

Runtime ส่งข้อมูลอย่างน้อย:

events (counted customer, sit/wait/wash, timestamp)

heartbeat/status (online/offline, last_seen, cameras_ok)

รองรับ retry/backoff เมื่อเน็ตหลุด

FR-07: Runtime Controls

ใน Controller มีปุ่ม:

Start / Stop runtime

Restart runtime

View runtime logs (tail)

Runtime ควรมี watchdog:

ถ้า RTSP หลุด ให้ reconnect

ถ้าหลุดนาน ให้แจ้งเตือนใน UI และส่ง status ไป Supabase

FR-08: Monitoring Dashboard (ภายใน Controller)

แสดงสถานะรวม:

Runtime: Running/Stopped

Cameras: OK/Fail รายตัว

Supabase: OK/Fail

Last heartbeat time

แสดงค่าตัวแปรสำคัญ (thresholds) และแก้ไขได้

4) Non-Functional Requirements (NFR)
NFR-01: ความเสถียร

Runtime ต้องทำงานต่อเนื่อง 24/7

เมื่อสตรีมหลุด ต้อง recover ได้อัตโนมัติ

NFR-02: ประสิทธิภาพ

รองรับอย่างน้อย 2–4 กล้องต่อเครื่อง (ขึ้นกับสเปค)

ควบคุม FPS per camera ได้ (เช่น 8–15)

NFR-03: ความปลอดภัย

หลีกเลี่ยงใช้ SUPABASE_SERVICE_ROLE_KEY บนเครื่องสาขา (ถ้าไม่จำเป็น)

เก็บ key แบบเข้ารหัสหรืออย่างน้อยจำกัดสิทธิ์ด้วย RLS/Policy

Log ต้องไม่ dump secrets

NFR-04: การติดตั้งและอัปเดต

ติดตั้งง่ายแบบ Setup.exe

มีเวอร์ชันโปรแกรม + โครงรองรับ auto-update (phase ต่อไป)

NFR-05: การใช้งาน

คนหน้าร้านใช้งานได้: มี Wizard + ปุ่ม Test ชัดเจน

Error message ต้องอ่านรู้เรื่อง และบอกวิธีแก้

5) Constraints / Assumptions

กล้องเป็น RTSP (H.264/H.265)

สภาพเน็ตมีโอกาสหลุด → ต้อง offline tolerant

ระบบต้องเก็บ logs/snapshots เพื่อ audit ความแม่นยำ

OS เป้าหมาย: (เลือก) Windows เป็นหลัก / macOS รอง

6) Deliverables (สิ่งที่ต้องส่งมอบ)

Controller App (GUI)

Runtime Service

Installer (Setup.exe) + คู่มือ

โครง config + ตัวอย่าง zones + ตัวอย่าง staff_gallery

Test plan + checklist

เอกสารการ deploy ต่อสาขา