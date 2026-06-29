"""Per-machine device identity that survives a reinstall of the project folder.

Stored at %LOCALAPPDATA%\\HGCameraCounter\\device_identity.json — OUTSIDE the project
directory (which a reinstall / re-clone wipes). The device_token is DPAPI machine-bound
(same scheme as config secrets, see shared/secure.py), so the file is useless if copied
to another PC.

Purpose: when a branch PC is reinstalled or re-paired, it keeps its ORIGINAL device code
+ token instead of creating a duplicate device — i.e. "remember the device number".
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict

try:
    from shared.secure import encrypt_secret, decrypt_secret, is_encrypted
except Exception:  # pragma: no cover - fallback for odd import contexts
    try:
        from .secure import encrypt_secret, decrypt_secret, is_encrypted
    except Exception:
        encrypt_secret = decrypt_secret = is_encrypted = None

# Fields encrypted at rest (machine-bound). Everything else is plain (codes/names).
_SECRET_FIELDS = ("device_token",)


def identity_path() -> Path:
    base = os.environ.get("LOCALAPPDATA") or os.environ.get("APPDATA") or str(Path.home())
    return Path(base) / "HGCameraCounter" / "device_identity.json"


def save_identity(data: Dict) -> bool:
    """Persist the device identity (device_code/name/token/branch_code/timezone).
    Token is DPAPI-encrypted. Returns True on success."""
    try:
        out = {k: v for k, v in (data or {}).items() if v not in (None, "")}
        if not out:
            return False
        if encrypt_secret and is_encrypted:
            for f in _SECRET_FIELDS:
                v = out.get(f)
                if v and not is_encrypted(v):
                    try:
                        out[f] = encrypt_secret(v)
                    except Exception:
                        pass
        p = identity_path()
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
        return True
    except Exception:
        return False


def load_identity() -> Dict:
    """Return the stored identity with the token decrypted to plaintext, or {}."""
    try:
        p = identity_path()
        if not p.exists():
            return {}
        data = json.loads(p.read_text(encoding="utf-8")) or {}
        if not isinstance(data, dict):
            return {}
        if decrypt_secret and is_encrypted:
            for f in _SECRET_FIELDS:
                v = data.get(f)
                if v and is_encrypted(v):
                    try:
                        data[f] = decrypt_secret(v)
                    except Exception:
                        data[f] = ""  # cannot decrypt (e.g. copied from another PC)
        return data
    except Exception:
        return {}


def clear_identity() -> bool:
    try:
        p = identity_path()
        if p.exists():
            p.unlink()
        return True
    except Exception:
        return False
