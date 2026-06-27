# คู่มือติดตั้ง HG Camera Counter (เครื่องสาขาใหม่)

ติดตั้งบน Windows 10/11 ด้วยคำสั่งเดียว ตัว `setup.bat` จะลง Python 3.11 + Git +
ไลบรารีทั้งหมด + ffmpeg + ตั้งให้เปิด/นับเองตอนบูตให้อัตโนมัติ

---

## ขั้นที่ 1 — เอาโค้ดลงเครื่อง

เลือกวิธีใดวิธีหนึ่ง (repo เป็นส่วนตัว ต้อง login GitHub):

**A) โหลด ZIP (ไม่ต้องมี Git)**
1. เปิดเบราว์เซอร์ login GitHub → ไปที่ repo (branch `main`)
2. ปุ่มเขียว **Code → Download ZIP**
3. แตกไฟล์ไปไว้ที่ เช่น `C:\HGCameraCounter`

**B) git clone** (ถ้ามี Git อยู่แล้ว)
```powershell
git clone https://github.com/huagruay-git/HGCameraCounter.git C:\HGCameraCounter
```

---

## ขั้นที่ 2 — รันตัวติดตั้ง

เข้าโฟลเดอร์โปรเจกต์ แล้ว **ดับเบิลคลิก `setup.bat`** (หรือพิมพ์ `.\setup.bat` ใน PowerShell)

จะติดตั้งให้อัตโนมัติ:
- ✅ Python 3.11 + Git (ผ่าน winget ถ้ายังไม่มี — อาจมี UAC เด้งให้กด Yes)
- ✅ สร้าง `.venv` + ลงไลบรารีทั้งหมด (torch ฯลฯ — ใช้เวลาหลายนาที)
- ✅ ffmpeg
- ✅ shortcut เปิด+นับเองตอนล็อกอิน

> เพิ่ม watchdog (รีบูตเองถ้าโปรแกรมค้าง) ด้วย: `.\setup.bat -Watchdog` (รัน as administrator)

---

## ขั้นที่ 3 — ใส่ข้อมูลเฉพาะเครื่อง (สำคัญ ต้องทำเอง)

**3.1 โมเดล AI**
- คัดลอก `models\best.pt` จากเครื่องเดิมมาวางที่ `models\best.pt`
- หรือ (เมื่อบริษัทอัปโมเดลขึ้น Supabase แล้ว) เปิดโปรแกรม → แท็บ **อัปเดตโมเดล** → เลือกเวอร์ชัน → ดาวน์โหลด

**3.2 ตั้งค่า + จับคู่อุปกรณ์** — เปิดโปรแกรมครั้งแรก:
```powershell
.venv\Scripts\python.exe controller\main.py
```
- ตั้ง **PIN** (ครั้งแรก — ผูกกับเครื่องนี้)
- แท็บ **Cloud Sync**: จับคู่อุปกรณ์กับ HQ (ได้ device token) + ใส่ Supabase URL/key + `branch_code` ของสาขา
- แท็บ **Cameras**: ใส่กล้อง (IP/RTSP) ของสาขานั้น

**3.3 เข้ารหัส secret** (กันข้อมูลรั่วถ้าเครื่องหาย):
```powershell
.venv\Scripts\python.exe scripts\encrypt_config_secrets.py
```

> ⚠️ ห้ามก๊อป `data\config\config.yaml` ที่เข้ารหัสแล้วจากเครื่องอื่นมาใช้ตรงๆ —
> secret ผูกกับเครื่อง (DPAPI) เครื่องใหม่ถอดรหัสไม่ออก ต้องกรอกใหม่แล้วเข้ารหัสที่เครื่องนั้น

---

## ขั้นที่ 4 — ตั้งให้เปิดเองตอนไฟมา (ที่ตัวเครื่อง)

1. **BIOS** → `Restore on AC Power Loss` = **Power On** (ไฟมาแล้วเครื่องติดเอง)
2. **Windows auto-login** → ใช้ [Sysinternals Autologon](https://learn.microsoft.com/sysinternals/downloads/autologon) หรือ `netplwiz` (เข้า Windows เองไม่ต้องใส่รหัส)

---

## ขั้นที่ 5 — ทดสอบ

**รีสตาร์ทเครื่อง 1 ครั้ง** → โปรแกรมควรเปิดเอง + เริ่มนับเองโดยไม่ต้องแตะอะไร ✅

---

## สรุปสั้น
```
โหลด ZIP/clone (main) → ดับเบิลคลิก setup.bat
→ วาง best.pt + ตั้ง PIN/Supabase/กล้อง + encrypt
→ BIOS + auto-login → รีสตาร์ททดสอบ
```

## แก้ปัญหาที่พบบ่อย
- **setup.bat ลง Python ไม่ได้** → ติดตั้งเอง [Python 3.11](https://www.python.org/downloads/release/python-3119/) (ติ๊ก Add to PATH) แล้วรัน `setup.bat` ใหม่
- **เปิดโปรแกรมแล้วค้างหน้า PIN ตอนบูต** → ปกติ; ตอน autostart จะข้าม PIN ให้เองถ้าเป็นเครื่องที่ตั้ง PIN+ผูกเครื่องไว้แล้ว (ขั้น 3.2)
- **ปิด autostart** → `powershell -ExecutionPolicy Bypass -File scripts\install_autostart.ps1 -Uninstall`
- **เครื่องไม่มีเน็ตตอนติดตั้ง** → ต้องมีเน็ต (โหลด Python/ไลบรารี/ffmpeg) อย่างน้อยตอน setup ครั้งแรก
