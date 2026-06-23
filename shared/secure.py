"""
Local security primitives for HG Camera Counter (Windows-first, no extra deps).

Three independent pieces, all self-describing so values survive a YAML round-trip:

  * PIN gate        ->  hash_pin / verify_pin  (PBKDF2-HMAC-SHA256, salted)
                        stored as "pbkdf2$<iters>$<salt_b64>$<hash_b64>"
  * device binding  ->  machine_fingerprint()  (Windows MachineGuid + hostname, hashed)
  * secrets at rest ->  encrypt_secret / decrypt_secret  (Windows DPAPI, machine scope)
                        stored as "enc:dpapi:<ciphertext_b64>"

DPAPI uses the LOCAL_MACHINE scope, so a config file copied to a *different*
machine can no longer be decrypted -- that is the real "device pairing": the
anon key / device_token become useless off the registered box. The PIN is the
second factor a human must type to open the app.

Everything degrades gracefully on non-Windows (encryption becomes a no-op) so the
module stays importable for tests/tooling on other platforms.
"""
from __future__ import annotations

import base64
import ctypes
import hashlib
import hmac
import os
import platform

ENC_PREFIX = "enc:dpapi:"
PIN_PREFIX = "pbkdf2$"
_PBKDF2_ITERS = 200_000

# CryptProtectData flags
_CRYPTPROTECT_UI_FORBIDDEN = 0x1
_CRYPTPROTECT_LOCAL_MACHINE = 0x4
_DPAPI_FLAGS = _CRYPTPROTECT_LOCAL_MACHINE | _CRYPTPROTECT_UI_FORBIDDEN


# --------------------------------------------------------------------------- #
# PIN hashing (login gate)
# --------------------------------------------------------------------------- #
def hash_pin(pin: str, *, iterations: int = _PBKDF2_ITERS) -> str:
    """Return a salted PBKDF2 hash string for `pin` (store this, never the PIN)."""
    salt = os.urandom(16)
    dk = hashlib.pbkdf2_hmac("sha256", str(pin).encode("utf-8"), salt, iterations)
    return f"{PIN_PREFIX}{iterations}${base64.b64encode(salt).decode()}${base64.b64encode(dk).decode()}"


def verify_pin(pin: str, stored: str) -> bool:
    """Constant-time check of `pin` against a hash from hash_pin()."""
    try:
        if not stored or not stored.startswith(PIN_PREFIX):
            return False
        _, iters_s, salt_b64, hash_b64 = stored.split("$", 3)
        salt = base64.b64decode(salt_b64)
        expected = base64.b64decode(hash_b64)
        dk = hashlib.pbkdf2_hmac("sha256", str(pin).encode("utf-8"), salt, int(iters_s))
        return hmac.compare_digest(dk, expected)
    except Exception:
        return False


# --------------------------------------------------------------------------- #
# Machine fingerprint (device binding)
# --------------------------------------------------------------------------- #
def _windows_machine_guid() -> str:
    try:
        import winreg
        with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE,
                            r"SOFTWARE\Microsoft\Cryptography") as k:
            val, _ = winreg.QueryValueEx(k, "MachineGuid")
            return str(val)
    except Exception:
        return ""


def machine_fingerprint() -> str:
    """Stable, privacy-preserving per-machine id (hex, 32 chars).

    Derived from the Windows MachineGuid + hostname, then hashed so the raw
    identifiers are never stored. Used to pin a config to one physical box.
    """
    parts = [platform.node() or "", _windows_machine_guid()]
    raw = "|".join(p for p in parts if p)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:32]


# --------------------------------------------------------------------------- #
# DPAPI secret encryption (secrets at rest, machine-bound)
# --------------------------------------------------------------------------- #
def dpapi_available() -> bool:
    return os.name == "nt"


def is_encrypted(value) -> bool:
    return isinstance(value, str) and value.startswith(ENC_PREFIX)


if os.name == "nt":
    import ctypes.wintypes as _wt

    class _DATA_BLOB(ctypes.Structure):
        _fields_ = [("cbData", _wt.DWORD), ("pbData", ctypes.POINTER(ctypes.c_char))]

    _crypt32 = ctypes.WinDLL("crypt32", use_last_error=True)
    _kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    _BLOB_P = ctypes.POINTER(_DATA_BLOB)
    _crypt32.CryptProtectData.restype = _wt.BOOL
    _crypt32.CryptProtectData.argtypes = [_BLOB_P, _wt.LPCWSTR, _BLOB_P,
                                          ctypes.c_void_p, ctypes.c_void_p,
                                          _wt.DWORD, _BLOB_P]
    _crypt32.CryptUnprotectData.restype = _wt.BOOL
    _crypt32.CryptUnprotectData.argtypes = [_BLOB_P, ctypes.c_void_p, _BLOB_P,
                                            ctypes.c_void_p, ctypes.c_void_p,
                                            _wt.DWORD, _BLOB_P]
    _kernel32.LocalFree.restype = _wt.HLOCAL
    _kernel32.LocalFree.argtypes = [_wt.HLOCAL]

    def _dpapi(func, data: bytes) -> bytes:
        # Keep `buf` alive for the whole call: the input blob holds a raw pointer
        # into it, so it must not be GC'd before the API returns.
        buf = ctypes.create_string_buffer(data, len(data))
        blob_in = _DATA_BLOB(len(data), ctypes.cast(buf, ctypes.POINTER(ctypes.c_char)))
        blob_out = _DATA_BLOB()
        ok = func(ctypes.byref(blob_in), None, None, None, None,
                  _DPAPI_FLAGS, ctypes.byref(blob_out))
        if not ok:
            raise OSError(f"DPAPI call failed (err={ctypes.get_last_error()})")
        try:
            return ctypes.string_at(blob_out.pbData, blob_out.cbData)
        finally:
            _kernel32.LocalFree(ctypes.cast(blob_out.pbData, _wt.HLOCAL))


def encrypt_secret(plaintext: str) -> str:
    """DPAPI-encrypt a secret -> 'enc:dpapi:...'. No-op if empty/already-encrypted.

    On non-Windows this returns the value unchanged (callers should warn).
    """
    if not plaintext or is_encrypted(plaintext):
        return plaintext
    if not dpapi_available():
        return plaintext
    enc = _dpapi(_crypt32.CryptProtectData, str(plaintext).encode("utf-8"))
    return ENC_PREFIX + base64.b64encode(enc).decode("ascii")


def decrypt_secret(value: str):
    """Decrypt an 'enc:dpapi:...' value. Pass-through for anything else.

    Raises OSError if the blob was produced on another machine (anti-theft).
    """
    if not is_encrypted(value):
        return value
    if not dpapi_available():
        return value
    raw = base64.b64decode(value[len(ENC_PREFIX):])
    dec = _dpapi(_crypt32.CryptUnprotectData, raw)
    return dec.decode("utf-8")


if __name__ == "__main__":
    # Smoke test on the local machine.
    h = hash_pin("123456")
    print("pin ok:", verify_pin("123456", h), "| pin bad:", verify_pin("000000", h))
    print("machine:", machine_fingerprint())
    if dpapi_available():
        e = encrypt_secret("super-secret-token")
        print("enc:", e[:32], "...")
        print("roundtrip ok:", decrypt_secret(e) == "super-secret-token")
    else:
        print("DPAPI not available on this platform (no-op mode)")
