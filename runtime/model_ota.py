"""Model + config OTA client (device-token authenticated).

Pulls the active config + active YOLO model for this device's branch from Supabase,
applies config live through the runtime override file, and swaps the local model
file when a newer version is published (verified by sha256). Model reload takes
effect on the next runtime restart.

Prerequisite: deploy the device-OTA RPCs first —
  supabase/migrations/20260616_cctv_device_ota.sql
and upload a model to the `yolo-models` bucket + mark a yolo_models row is_active.

Run standalone:
  python runtime/model_ota.py            # uses data/config/config.yaml
  python runtime/model_ota.py --dry-run  # check only, do not download/swap
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import tempfile
import urllib.request
from pathlib import Path
from typing import Optional


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


class ModelOTA:
    def __init__(self, rpc_client, device_token: str, local_model_path, override_path,
                 logger=None, dry_run: bool = False):
        self.rpc = rpc_client
        self.device_token = (device_token or "").strip()
        self.local_model_path = Path(local_model_path)
        self.override_path = Path(override_path)
        self.logger = logger
        self.dry_run = bool(dry_run)
        self.version_marker = self.local_model_path.with_suffix(
            self.local_model_path.suffix + ".version")

    def _log(self, level: str, msg: str) -> None:
        if self.logger is not None:
            getattr(self.logger, level, self.logger.info)(msg)
        else:
            print(f"[{level.upper()}] {msg}")

    def _local_version(self) -> str:
        try:
            return self.version_marker.read_text(encoding="utf-8").strip()
        except Exception:
            return ""

    def _apply_config(self, config) -> bool:
        if not isinstance(config, dict) or not config:
            return False
        if self.dry_run:
            self._log("info", f"OTA(dry-run): would apply {len(config)} config keys")
            return False
        try:
            self.override_path.parent.mkdir(parents=True, exist_ok=True)
            existing = {}
            if self.override_path.exists():
                try:
                    existing = json.loads(self.override_path.read_text(encoding="utf-8")) or {}
                except Exception:
                    existing = {}
            existing.update(config)
            self.override_path.write_text(
                json.dumps(existing, ensure_ascii=False, indent=2), encoding="utf-8")
            self._log("info", f"OTA: wrote {len(config)} config key(s) to override (hot-apply)")
            return True
        except Exception as e:
            self._log("error", f"OTA: failed to apply config: {e}")
            return False

    def _download(self, model: dict, dest: Path) -> None:
        signed = model.get("signed_url")
        if signed:
            with urllib.request.urlopen(signed, timeout=180) as r, open(dest, "wb") as fd:
                shutil.copyfileobj(r, fd)
            return
        bucket = model.get("storage_bucket") or "yolo-models"
        path = model.get("storage_path")
        if not path:
            raise ValueError("model entry has no signed_url and no storage_path")
        data = self.rpc.download_storage_object(bucket, path)
        with open(dest, "wb") as fd:
            fd.write(data if isinstance(data, (bytes, bytearray)) else bytes(data))

    def _apply_model(self, model: dict) -> bool:
        version = str(model.get("model_version") or "").strip()
        if not version:
            return False
        if version == self._local_version() and self.local_model_path.exists():
            self._log("info", f"OTA: model already at version {version}; skip")
            return False
        if self.dry_run:
            self._log("info", f"OTA(dry-run): new model {version} available (local={self._local_version() or 'none'})")
            return False

        tmpdir = Path(tempfile.mkdtemp(prefix="model_ota_"))
        tmp = tmpdir / self.local_model_path.name
        sha_ok: Optional[bool] = None
        try:
            self._download(model, tmp)
            expected = (model.get("sha256") or "").strip()
            if expected:
                sha_ok = _sha256(tmp).lower() == expected.lower()
                if not sha_ok:
                    raise ValueError("sha256 mismatch (download rejected)")
            self.local_model_path.parent.mkdir(parents=True, exist_ok=True)
            if self.local_model_path.exists():
                shutil.copy2(self.local_model_path,
                             self.local_model_path.with_suffix(self.local_model_path.suffix + ".bak"))
            shutil.move(str(tmp), str(self.local_model_path))
            self.version_marker.write_text(version, encoding="utf-8")
            self._log("info", f"OTA: model updated -> {version} (active on next runtime restart)")
            self._safe_log(model, "success", f"updated to {version}", sha_ok)
            return True
        except Exception as e:
            self._log("error", f"OTA: model update failed: {e}")
            self._safe_log(model, "failed", str(e), sha_ok)
            return False
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

    def _safe_log(self, model: dict, status: str, message: str, sha_ok) -> None:
        try:
            self.rpc.log_cctv_model_download(
                self.device_token, model.get("id"), model.get("model_version"),
                status, message, sha_ok)
        except Exception as e:
            self._log("warning", f"OTA: could not write model_download_log: {e}")

    def check_and_apply(self) -> dict:
        result = {"config_applied": False, "model_updated": False, "error": None}
        try:
            bs = self.rpc.get_cctv_runtime_bootstrap(self.device_token) or {}
            result["config_applied"] = self._apply_config(bs.get("config"))
            model = bs.get("model")
            if isinstance(model, dict) and model:
                result["model_updated"] = self._apply_model(model)
            else:
                self._log("info", "OTA: no active model returned for this branch")
        except Exception as e:
            result["error"] = str(e)
            self._log("error", f"OTA: bootstrap failed: {e}")
        return result


def _load_yaml(path: Path) -> dict:
    import yaml
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    # Decrypt machine-bound secrets (anon key / device_token) — this module reads
    # the config raw instead of via Config, so do the same transparent decrypt.
    try:
        from shared.config import _decrypt_secrets
        data = _decrypt_secrets(data)
    except Exception:
        pass
    return data


def main() -> int:
    parser = argparse.ArgumentParser(description="Model/config OTA pull from Supabase")
    parser.add_argument("--config", default="data/config/config.yaml")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    sys.path.insert(0, os.path.abspath("."))
    from shared.logger import setup_logger
    from shared.supabase_client import CCTVSupabaseRPCClient

    cfg = _load_yaml(Path(args.config))
    sb = cfg.get("supabase", {}) or {}
    cloud = sb.get("cloud_sync", {}) or {}
    paths = cfg.get("paths", {}) or {}
    yolo = cfg.get("yolo", {}) or {}

    url = sb.get("url", "")
    key = sb.get("key", "")
    device_token = cloud.get("device_token", "")
    models_dir = paths.get("models", "models")
    model_file = yolo.get("worker_model") or yolo.get("model") or "best.pt"
    local_model = Path(models_dir) / model_file
    override = paths.get("runtime_settings_override", "runtime/runtime_settings.override.json")

    logger = setup_logger("model_ota", paths.get("logs", "logs"))
    if not (url and key and device_token):
        logger.error("OTA: missing supabase url/key/device_token in config")
        return 2

    rpc = CCTVSupabaseRPCClient(url, key, logger=logger)
    ota = ModelOTA(rpc, device_token, local_model, override, logger=logger, dry_run=args.dry_run)
    result = ota.check_and_apply()
    logger.info(f"OTA result: {result}")
    print(json.dumps(result, ensure_ascii=False))
    return 0 if not result.get("error") else 1


if __name__ == "__main__":
    raise SystemExit(main())
