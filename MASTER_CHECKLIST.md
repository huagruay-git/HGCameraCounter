# Master Checklist (สิ่งที่ต้องทำทั้งหมด)

> สถานะ: [ ] = ยังไม่ทำ, [x] = เสร็จแล้ว

---

## A) Project Setup ✅ COMPLETED
- [x] สร้าง repo โครงสร้างโปรเจกต์ (controller / runtime / shared)
- [x] กำหนด format config กลาง (config.json หรือ yaml)
- [x] ตั้ง logging มาตรฐาน (INFO/WARN/ERROR + rotating)
- [x] ทำ folder convention: reports/, snapshots/, staff_gallery/, zones/, models/, logs/

---

## B) Controller App (GUI)

### B1: หน้าหลัก + โครง UI
- [x] เลือก Framework GUI (PySide6) ✅
- [x] หน้า Home: สถานะรวม + ปุ่ม Start/Stop/Restart ✅
- [x] เมนู/แท็บ: Setup Wizard / Cameras / Zones / Staff DB / Diagnostics / Logs / Settings ✅

### B2: Setup Wizard ✅ PARTIAL
- [x] Step 1: Supabase Settings + Test
- [x] Step 2: Camera Add/Edit/Delete + Test RTSP
- [x] Step 3: Zone Editor (load snapshot/live + draw polygon)
- [x] Step 4: Staff Gallery + Build staff_db.json
- [x] Step 5: Diagnostics Summary (Pass/Fail รายหัวข้อ)
- [ ] Step 6: Deploy/Install Service + Run

### B3: Dashboard & Real-time Updates ✅ COMPLETE [Phase 3A]
- [x] Real-time camera status display (connection + FPS)
- [x] Live event counts (haircuts/wash/wait with timestamp)
- [x] Active people tracking (live count from tracker)
- [x] Connection status indicator (🟢 Live / ⚠️ No connection)
- [x] Auto-refresh every 2-5 seconds
- [x] Manual refresh button
- [x] Status/event/summary signal handlers

### B4: Camera Management ✅ COMPLETE
- [x] Form เพิ่มกล้อง (name, rtsp_url, enabled, note)
- [x] ปุ่ม Test RTSP (connect + frame grab + snapshot)
- [x] Preview ภาพ + แสดงค่า latency/fps โดยประมาณ
- [x] Save/Load กล้องเข้า config
- [x] Import/Export รายการกล้อง (ไฟล์ json)

### B5: Zone Editor
- [x] โหลดภาพจากกล้อง (snapshot หรือ live frame)
- [x] เครื่องมือวาด polygon + edit จุด (drag/add/remove)
- [x] ตั้งชื่อโซน + ประเภทโซน (CHAIR/WAIT/WASH/STAFF_AREA/OTHER)
- [x] Save/Load zones_*.json ต่อกล้อง
- [x] Validate polygon (>=3 จุด, ไม่ว่าง, อยู่ในขอบภาพ)
- [x] (Optional) ตรวจ overlap ระหว่างโซน (แจ้งเตือน)

### B6: Staff DB Builder UI
- [x] เลือกโฟลเดอร์ staff_gallery
- [x] Scan staff folders + count images ต่อคน
- [x] ตรวจคุณภาพขั้นต่ำ (count/size/blur) + แนะนำเพิ่มรูป
- [x] ปุ่ม Build → สร้าง staff_db.json
- [x] รายงานผล success/fail รายรูป + เหตุผล
- [x] (Optional) save_crops สำหรับตรวจสอบ

### B7: Diagnostics UI (Health Checks)
- [x] Network check (DNS/Ping/Speed แบบเบา)
- [x] RTSP check per camera (OK/Fail + reason)
- [x] Model check (ไฟล์ YOLO + staff_db + zones)
- [x] Disk/permission check (write snapshots/reports/logs)
- [x] Device/GPU check (cpu/mps/cuda + fps estimate)
- [x] Supabase check (connect + identify branch)
- [x] สรุปผลเป็นแผงเดียว + export รายงาน

### B8: Logs Viewer
- [x] แสดง log ล่าสุดแบบ tail
- [x] กรองตาม level/camera
- [x] ปุ่ม Export log
- [x] ปุ่ม Open logs folder

---

## C) Runtime Service ✅ COMPLETE (100%) [Phase 3A & 3B]
- [x] โหลด config กลาง + zones + staff_db
- [x] จัดการ multi-camera pipeline (thread/process)
- [x] YOLO detect + tracking + zone dwell logic
- [x] Logic นับเหตุการณ์ haircut / wait / wash ตาม dwell time
- [x] บันทึก reports CSV / snapshots / daily summary
- [x] ส่ง event เข้า Supabase (retry/backoff)
- [x] Heartbeat ส่งสถานะ (online/offline, cameras_ok, last_seen) ทุก N วินาที
- [x] Watchdog reconnect RTSP (Phase 3B - auto-reconnect with exponential backoff)
- [x] Graceful shutdown + restart safe
- [x] จำกัด resource (FPS cap, queue size, memory guard) (Phase 3B - Resource Guard)
- [x] HealthChecker - periodic diagnostics (disk/memory/cpu/network/permissions) (Phase 3B)

---

## D) Supabase (Backend) ✅ COMPLETE (70%)
- [x] ออกแบบตาราง device_status (heartbeat)
- [x] ออกแบบตาราง events หรือ counts (raw events + daily aggregates)
- [x] กำหนด RLS/Policy ให้ปลอดภัย (branch-scoped)
- [x] ทำ endpoint/test query สำหรับ "เช็คสาขา (branch_code)"
- [ ] ทำ RPC ping/health (optional) เพื่อ health check ง่าย ๆ
- [ ] ทำระบบแจ้งเตือนเมื่อ offline (optional/phase ต่อไป)

---

## E) Packaging / Installer
- [x] สร้าง build script (PyInstaller) + spec แยก Controller/Runtime
- [x] รวม assets สำคัญใน build (data/, models/, runtime/, shared/, bytetrack.yaml)
- [x] ทำ Setup.exe ด้วย Inno Setup (`packaging/windows/HGCameraCounter.iss`)
- [ ] ทดสอบ build บนเครื่อง Windows จริงตามคู่มือ (`PACKAGING_WINDOWS.md`) ให้ได้ไฟล์ `dist/HGCameraCounter/HGCameraCounter.exe` และ `dist/runtime_service/runtime_service.exe`
- [ ] ติดตั้งเป็น Windows Service (nssm หรือ wrapper เช่น WinSW)
- [ ] ตั้ง auto-start + shortcut Controller และตรวจว่า GUI เรียก `runtime_service.exe` ข้างตัวเองได้
- [ ] ทดสอบ install/upgrade/uninstall + backup/restore config และทำ versioning/release notes

---

## F) QA / Testing
- [ ] Test RTSP: url ถูก/ผิด/timeout/credentials
- [ ] Test zone editor: save/load, polygon invalid
- [ ] Test staff_db: รูปน้อย, รูปเสีย, ไม่มีโฟลเดอร์
- [ ] Test offline mode: เน็ตหลุดแล้วกลับมา (queue + retry)
- [ ] Test multi-cam load: 1/2/4 กล้อง (fps/latency)
- [ ] Test Supabase permission: key จำกัดสิทธิ์จริง
- [ ] Test upgrade/reinstall ไม่พัง config เดิม
- [ ] End-to-end test ต่อสาขาจริง 1 สาขา (pilot)

---

## G) Documentation
- [ ] คู่มือ Setup ต่อสาขา (Step-by-step)
- [ ] คู่มือ Troubleshoot (RTSP, network, supabase, model)
- [ ] คู่มือเพิ่มพนักงาน/เพิ่มกล้อง/วาดโซน
- [ ] คู่มือดูสถานะออนไลน์/อ่าน report
- [ ] คู่มือ backup/restore และการอัปเดตเวอร์ชัน

---

## Summary

**Total Items**: 85

| Section | Items | Completed |
|---------|-------|-----------|
| A) Setup | 4 | 4 ✅ |
| B) Controller | 44 | 43 (97.7%) |
| C) Runtime | 11 | 11 ✅ (100%) |
| D) Supabase | 6 | 4 (66.7%) |
| E) Packaging | 7 | 3 (42.9%) |
| F) Testing | 8 | 0 |
| G) Documentation | 5 | 0 |

**Overall**: 65/85 items (76.5%) completed

---

## Phase 1 Status: ✅ COMPLETE

### Completed:
- ✅ Project structure created
- ✅ Centralized config system (YAML)
- ✅ Standard logging setup
- ✅ Runtime service refactored
- ✅ Controller GUI (Setup Wizard + Main App)
- ✅ requirements.txt created
- ✅ Documentation created

### Next: Phase 2 - Feature Implementation
- Runtime event counting logic
- Supabase integration
- Full UI features
- Packaging & installer

---

**Last Updated**: 2026-02-13 (Packaging checklist refined for Windows EXE)

Phase Status:
- Phase 1: ✅ Complete (Project Setup)
- Phase 2: ✅ Complete (Event Logic + Supabase)
- Phase 3A: ✅ Complete (Real-time Dashboard)
- Phase 3B: ✅ Complete (Reliability: RTSP Watchdog + Resource Guard + Health Checks)
- Phase 4: 🚀 Starting (B4 Camera Management + B5 Zone Editor + More UI)
