"""
Startup login gate: shared PIN + device pairing (machine binding).

- First run (no PIN yet): asks the operator to create a PIN, binds the app to this
  machine's fingerprint, and stores both in data/config/auth.json.
- Later runs: rejects the auth file if it was copied to another machine (fingerprint
  mismatch), then asks for the PIN with a limited number of attempts.

This is the human factor (PIN) + device factor (machine binding). The Supabase
secrets are separately DPAPI-bound to this machine (see shared/secure.py), so even a
full config copy is useless on another PC.

auth.json (data/config/, gitignored):
  {"pin_hash": "pbkdf2$...", "bound_machine": "<fp>", "created": <ts>}
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

from PySide6.QtWidgets import QInputDialog, QMessageBox, QLineEdit

from shared.secure import hash_pin, verify_pin, machine_fingerprint

AUTH_PATH = Path(__file__).resolve().parent.parent / "data" / "config" / "auth.json"
MAX_ATTEMPTS = 5
MIN_PIN_LEN = 4


def _load_auth() -> dict:
    try:
        return json.loads(AUTH_PATH.read_text(encoding="utf-8")) or {}
    except Exception:
        return {}


def _save_auth(auth: dict) -> None:
    AUTH_PATH.parent.mkdir(parents=True, exist_ok=True)
    AUTH_PATH.write_text(json.dumps(auth, ensure_ascii=False, indent=2), encoding="utf-8")


def _ask_pin(title: str, label: str):
    text, ok = QInputDialog.getText(None, title, label, QLineEdit.Password)
    if not ok:
        return None
    return (text or "").strip()


def _first_run_setup():
    QMessageBox.information(
        None, "ตั้งค่าครั้งแรก",
        "ยังไม่ได้ตั้ง PIN สำหรับเครื่องนี้\nกรุณาตั้ง PIN เพื่อยืนยันสิทธิ์การเข้าใช้งาน")
    while True:
        pin = _ask_pin("ตั้ง PIN", f"ตั้ง PIN ใหม่ (อย่างน้อย {MIN_PIN_LEN} ตัว):")
        if pin is None:
            return None
        if len(pin) < MIN_PIN_LEN:
            QMessageBox.warning(None, "PIN สั้นไป", f"PIN ต้องยาวอย่างน้อย {MIN_PIN_LEN} ตัว")
            continue
        confirm = _ask_pin("ยืนยัน PIN", "พิมพ์ PIN อีกครั้ง:")
        if confirm is None:
            return None
        if pin != confirm:
            QMessageBox.warning(None, "ไม่ตรงกัน", "PIN สองครั้งไม่ตรงกัน ลองใหม่")
            continue
        auth = {
            "pin_hash": hash_pin(pin),
            "bound_machine": machine_fingerprint(),
            "created": int(time.time()),
        }
        _save_auth(auth)
        QMessageBox.information(None, "สำเร็จ", "ตั้ง PIN และผูกเครื่องเรียบร้อยแล้ว")
        return auth


def _is_autostart() -> bool:
    """True when the app was launched unattended at boot (HGCC_AUTOSTART=1).

    Set by the Startup-folder launcher / the ``--autostart`` flag in controller.main.
    Used to skip the *interactive* PIN prompt on an already-provisioned, machine-bound
    device so a branch PC can resume counting after a power cut without a human. The
    device-binding check below is NEVER skipped, so a stolen/copied install is still
    refused (and its DPAPI secrets are undecryptable on another machine anyway).
    """
    return os.environ.get("HGCC_AUTOSTART", "").strip().lower() in ("1", "true", "yes")


def run_login_gate(config=None) -> bool:
    """Return True if the operator may open the app, else False (caller should exit)."""
    auth = _load_auth()
    autostart = _is_autostart()

    # Soft pairing note: a device with no HQ token is unprovisioned (not blocking).
    try:
        token = (config.get("supabase", {}) or {}).get("cloud_sync", {}).get("device_token") if config else None
        if not token:
            print("[login] WARNING: no device_token in config (device not paired with HQ).")
    except Exception:
        pass

    # First run -> create PIN + bind machine. Autostart cannot do interactive setup;
    # require a human to provision the device once before unattended boots will work.
    if not auth.get("pin_hash"):
        if autostart:
            print("[login] autostart: no PIN set yet — run once manually to set the PIN. Refusing unattended start.")
            return False
        return _first_run_setup() is not None

    # Device binding: reject an auth file copied from another machine (always enforced).
    fp = machine_fingerprint()
    bound = auth.get("bound_machine")
    if bound and bound != fp:
        if autostart:
            print("[login] autostart: machine fingerprint mismatch — refusing unattended start.")
            return False
        QMessageBox.critical(
            None, "เครื่องไม่ได้รับอนุญาต",
            "เครื่องนี้ไม่ใช่เครื่องที่ลงทะเบียนไว้\nกรุณาติดต่อสำนักงานใหญ่ (HQ)")
        return False

    # Autostart on the bound machine: device factor verified -> skip the human PIN.
    if autostart:
        print("[login] autostart: machine verified — skipping interactive PIN.")
        return True

    # PIN check with limited attempts.
    for attempt in range(1, MAX_ATTEMPTS + 1):
        remaining = MAX_ATTEMPTS - attempt + 1
        pin = _ask_pin("ใส่ PIN", f"ใส่ PIN เพื่อเข้าโปรแกรม (เหลือ {remaining} ครั้ง):")
        if pin is None:
            return False  # user cancelled
        if verify_pin(pin, auth["pin_hash"]):
            return True
        QMessageBox.warning(None, "PIN ไม่ถูกต้อง", "PIN ไม่ถูกต้อง")
    QMessageBox.critical(None, "ปฏิเสธการเข้าใช้งาน", "ใส่ PIN ผิดเกินจำนวนครั้งที่กำหนด")
    return False
