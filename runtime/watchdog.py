"""
Local liveness watchdog — auto-recovers a frozen / dead counting runtime by
rebooting the PC when the live agent stops refreshing its liveness signal.

WHY
  The edge mini-PC at a branch must keep counting 24/7 unattended. If the live
  runtime (runtime/agent_v2.py) hard-freezes (RTSP / GPU deadlock, OOM thrash)
  or the OS wedges, nothing recovers it without a human on site. This watchdog
  is a SEPARATE process (so it survives an app freeze), started at boot by
  Windows Task Scheduler, that watches the runtime's liveness file and issues
  `shutdown /r /f` when it goes stale — behind strong boot-loop guards.

LIVENESS SIGNAL: runtime/dashboard_state.json
  The live agent rewrites it (with a fresh `timestamp`) at least every ~10s
  while ANY of its threads is alive — the event-submission loop writes
  unconditionally on a 10s floor, the detection loop ~1s. Freshness is
  now - max(file mtime, inner "timestamp"). The default stale threshold (300s)
  sits ~30x above that cadence, so a healthy but IDLE salon (zero people, static
  counts) never trips it — only a real freeze / death does.

SAFETY (this process can reboot the machine)
  * opt-in            : watchdog.enabled defaults to false.
  * arm-on-seen       : by default only acts after observing a FRESH liveness at
                        least once, so a machine where counting was never started
                        is never rebooted.
  * missing != frozen : reboot_on_missing_file defaults false.
  * confirm           : N consecutive stale checks required (absorbs blips).
  * boot-loop guards  : min seconds between reboots + max reboots / 24h, persisted
                        across reboots; once exceeded the watchdog STOPS rebooting
                        and logs CRITICAL so a human investigates.
  * active hours      : optional window restricting when a reboot may fire.
  * dry_run           : evaluate + log the decision without actually rebooting.

Run
  python runtime/watchdog.py            # daemon (uses data/config/config.yaml)
  python runtime/watchdog.py --check    # print liveness age + decision, then exit
  python runtime/watchdog.py --once     # single evaluation, then exit
  python runtime/watchdog.py --dry-run  # never actually reboot (log only)
  HGCameraCounter.exe --watchdog        # frozen build (see packaging/launcher.py)
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Make `import shared.*` work when launched directly as a script (not just via
# the frozen launcher, which already puts the app dir on sys.path).
_THIS = Path(__file__).resolve()
_REPO_ROOT = _THIS.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from shared.config import _app_base  # noqa: E402  (path tweak must precede import)
from shared.logger import setup_logger  # noqa: E402


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
_DEFAULTS: Dict[str, Any] = {
    "enabled": False,
    "check_interval_sec": 30.0,
    "liveness_file": "",          # "" => paths.dashboard_state (runtime/dashboard_state.json)
    "stale_after_sec": 300.0,     # liveness age that means "frozen / dead"
    "boot_grace_sec": 180.0,      # ignore staleness for this long after watchdog start
    "reboot_on_missing_file": False,
    "require_seen_alive": True,   # only arm after observing a fresh liveness once
    "consecutive_stale_required": 2,
    "min_seconds_between_reboots": 900.0,   # boot-loop guard
    "max_reboots_per_day": 5,               # boot-loop guard (rolling 24h)
    "active_hours": "",           # "" => always; else "HH:MM-HH:MM" (wraps past midnight)
    "reboot_delay_sec": 30,       # shutdown /t — lets in-flight writes settle
    "reboot_message": "HGCC watchdog: counting runtime unresponsive - auto restart",
    "dry_run": False,
    "state_file": "data/state/watchdog_state.json",
}


def _as_float(v: Any, default: float) -> float:
    try:
        return float(v)
    except Exception:
        return default


def _as_int(v: Any, default: int) -> int:
    try:
        return int(v)
    except Exception:
        return default


def _as_bool(v: Any, default: bool) -> bool:
    if isinstance(v, bool):
        return v
    if isinstance(v, str):
        return v.strip().lower() in {"1", "true", "yes", "on"}
    if v is None:
        return default
    try:
        return bool(v)
    except Exception:
        return default


def _resolve(base: Path, rel: str) -> Path:
    p = Path(rel)
    return p if p.is_absolute() else (base / p).resolve()


def load_settings(config_path: Optional[str]) -> Tuple[Dict[str, Any], Path, Path, Path]:
    """Return (wd_cfg, liveness_path, logs_dir, base) — tolerant of a missing config."""
    base = _app_base()
    raw: Dict[str, Any] = {}
    cfg_file = _resolve(base, config_path or "data/config/config.yaml")
    if cfg_file.exists():
        try:
            import yaml
            raw = yaml.safe_load(cfg_file.read_text(encoding="utf-8")) or {}
        except Exception as e:  # never let a bad config stop the watchdog
            print(f"[watchdog] WARNING: could not read config ({e}); using defaults")

    wd = dict(_DEFAULTS)
    wd.update((raw.get("watchdog") or {}))

    paths = raw.get("paths") or {}
    liveness_rel = (wd.get("liveness_file") or "").strip() or \
        paths.get("dashboard_state", "runtime/dashboard_state.json")
    liveness_path = _resolve(base, liveness_rel)
    logs_dir = _resolve(base, paths.get("logs", "logs"))
    return wd, liveness_path, logs_dir, base


# ---------------------------------------------------------------------------
# Liveness
# ---------------------------------------------------------------------------
def _read_inner_timestamp(path: Path) -> Optional[float]:
    """Best-effort read of the `timestamp` field; tolerant of a mid-replace file."""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        ts = data.get("timestamp") if isinstance(data, dict) else None
        return float(ts) if ts is not None else None
    except Exception:
        return None


def liveness_age_sec(path: Path, now: float) -> Optional[float]:
    """Seconds since the liveness file was last fresh, or None if it does not exist."""
    try:
        mtime = path.stat().st_mtime
    except FileNotFoundError:
        return None
    except Exception:
        return None
    inner = _read_inner_timestamp(path)
    freshest = mtime if inner is None else max(mtime, inner)
    return max(0.0, now - freshest)


# ---------------------------------------------------------------------------
# Active-hours window
# ---------------------------------------------------------------------------
def within_active_hours(now_dt: datetime, window: str) -> bool:
    window = (window or "").strip()
    if not window:
        return True
    try:
        a, b = window.split("-", 1)
        sh, sm = (int(x) for x in a.strip().split(":"))
        eh, em = (int(x) for x in b.strip().split(":"))
        start = sh * 60 + sm
        end = eh * 60 + em
        cur = now_dt.hour * 60 + now_dt.minute
        if start == end:
            return True
        if start < end:
            return start <= cur < end
        return cur >= start or cur < end  # wraps past midnight
    except Exception:
        return True  # a malformed window must never block recovery


# ---------------------------------------------------------------------------
# Reboot history (persisted across reboots for the boot-loop guard)
# ---------------------------------------------------------------------------
class RebootHistory:
    def __init__(self, path: Path):
        self.path = path
        self.reboots: List[float] = []
        self._load()

    def _load(self) -> None:
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
            self.reboots = [float(x) for x in data.get("reboots", []) if x]
        except Exception:
            self.reboots = []

    def recent(self, now: float, window_sec: float = 86400.0) -> List[float]:
        return [t for t in self.reboots if (now - t) <= window_sec]

    def last(self) -> Optional[float]:
        return max(self.reboots) if self.reboots else None

    def record(self, now: float, reason: str) -> None:
        self.reboots.append(now)
        self.reboots = self.recent(now)  # prune > 24h
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text(
                json.dumps(
                    {
                        "reboots": self.reboots,
                        "last_reason": reason,
                        "updated_at": datetime.now().isoformat(timespec="seconds"),
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )
        except Exception:
            pass  # persistence is best-effort; the in-memory list still guards this run


# ---------------------------------------------------------------------------
# Watchdog
# ---------------------------------------------------------------------------
class Watchdog:
    def __init__(self, wd: Dict[str, Any], liveness_path: Path, base: Path, logger,
                 dry_run_override: bool = False):
        self.wd = wd
        self.liveness_path = liveness_path
        self.base = base
        self.log = logger

        self.check_interval = max(5.0, _as_float(wd.get("check_interval_sec"), 30.0))
        self.stale_after = max(30.0, _as_float(wd.get("stale_after_sec"), 300.0))
        self.boot_grace = max(0.0, _as_float(wd.get("boot_grace_sec"), 180.0))
        self.reboot_on_missing = _as_bool(wd.get("reboot_on_missing_file"), False)
        self.require_seen_alive = _as_bool(wd.get("require_seen_alive"), True)
        self.consecutive_required = max(1, _as_int(wd.get("consecutive_stale_required"), 2))
        self.min_between = max(0.0, _as_float(wd.get("min_seconds_between_reboots"), 900.0))
        self.max_per_day = max(1, _as_int(wd.get("max_reboots_per_day"), 5))
        self.active_hours = str(wd.get("active_hours", "") or "")
        self.reboot_delay = max(0, _as_int(wd.get("reboot_delay_sec"), 30))
        self.reboot_message = str(wd.get("reboot_message") or _DEFAULTS["reboot_message"])
        self.dry_run = dry_run_override or _as_bool(wd.get("dry_run"), False)

        self.history = RebootHistory(_resolve(base, str(wd.get("state_file") or _DEFAULTS["state_file"])))
        self.start_ts = time.time()
        self.armed = False
        self.stale_streak = 0

    # -- evaluation ---------------------------------------------------------
    def evaluate(self, now: Optional[float] = None) -> Dict[str, Any]:
        """Return a decision dict; does NOT act. {action, reason, age, ...}"""
        now = time.time() if now is None else now
        age = liveness_age_sec(self.liveness_path, now)

        if age is None:
            if not self.reboot_on_missing:
                self.stale_streak = 0
                return {"action": "noop", "reason": "liveness file missing (reboot_on_missing_file=false)", "age": None}
            stale = True
            age_display: Optional[float] = None
        else:
            age_display = age
            if age <= self.stale_after:
                self.armed = True
                self.stale_streak = 0
                return {"action": "healthy", "reason": "liveness fresh", "age": age}
            stale = True

        # --- stale path ---
        if self.require_seen_alive and not self.armed:
            self.stale_streak = 0
            return {"action": "noop", "reason": "stale but never seen alive since start (require_seen_alive)", "age": age_display}

        if (now - self.start_ts) < self.boot_grace:
            return {"action": "noop", "reason": "within boot grace", "age": age_display}

        self.stale_streak += 1
        if self.stale_streak < self.consecutive_required:
            return {"action": "warn", "reason": f"stale ({self.stale_streak}/{self.consecutive_required} confirms)", "age": age_display}

        # confirmed stale — apply reboot guards
        if not within_active_hours(datetime.now(), self.active_hours):
            return {"action": "hold", "reason": f"stale but outside active_hours={self.active_hours}", "age": age_display}

        last = self.history.last()
        if last is not None and (now - last) < self.min_between:
            return {"action": "suppressed",
                    "reason": f"reboot cooldown: last {int(now - last)}s ago < {int(self.min_between)}s",
                    "age": age_display}

        recent = self.history.recent(now)
        if len(recent) >= self.max_per_day:
            return {"action": "suppressed",
                    "reason": f"boot-loop guard: {len(recent)} reboots in 24h >= max {self.max_per_day} — HUMAN NEEDED",
                    "age": age_display}

        return {"action": "reboot", "reason": "liveness stale, guards passed", "age": age_display}

    # -- action -------------------------------------------------------------
    def _issue_reboot(self) -> None:
        if sys.platform != "win32":
            self.log.warning(f"[non-Windows] would run: shutdown /r /f /t {self.reboot_delay}")
            return
        try:
            subprocess.run(
                ["shutdown", "/r", "/f", "/t", str(self.reboot_delay), "/c", self.reboot_message[:127]],
                check=False,
            )
        except Exception as e:
            self.log.error(f"shutdown command failed: {e}")

    def act(self, decision: Dict[str, Any]) -> bool:
        """Carry out a decision. Returns True if a reboot was issued (loop should stop)."""
        action = decision["action"]
        age = decision.get("age")
        age_txt = "missing" if age is None else f"{age:.0f}s"

        if action in ("healthy",):
            self.log.debug(f"liveness OK (age={age_txt})")
            return False
        if action == "noop":
            self.log.debug(f"{decision['reason']} (age={age_txt})")
            return False
        if action == "warn":
            self.log.warning(f"{decision['reason']} (age={age_txt})")
            return False
        if action == "hold":
            self.log.warning(f"HOLD reboot — {decision['reason']} (age={age_txt})")
            return False
        if action == "suppressed":
            self.log.critical(f"SUPPRESSED reboot — {decision['reason']} (age={age_txt})")
            return False

        # action == "reboot"
        now = time.time()
        if self.dry_run:
            self.log.critical(f"[dry-run] WOULD REBOOT now — {decision['reason']} (age={age_txt})")
            self.stale_streak = 0
            return False

        self.history.record(now, decision["reason"])
        self.log.critical(
            f"REBOOTING (shutdown /r /f /t {self.reboot_delay}) — {decision['reason']} (age={age_txt})"
        )
        self._issue_reboot()
        return True

    # -- loop ---------------------------------------------------------------
    def run(self, once: bool = False) -> None:
        self.log.info(
            "watchdog started | liveness=%s stale_after=%.0fs interval=%.0fs dry_run=%s armed_required=%s"
            % (self.liveness_path, self.stale_after, self.check_interval, self.dry_run, self.require_seen_alive)
        )
        while True:
            try:
                decision = self.evaluate()
                rebooted = self.act(decision)
                if rebooted or once:
                    return
            except Exception as e:
                self.log.error(f"watchdog loop error: {e}")
            time.sleep(self.check_interval)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main() -> int:
    parser = argparse.ArgumentParser(description="HGCC local liveness watchdog")
    parser.add_argument("--config", default="data/config/config.yaml")
    parser.add_argument("--check", action="store_true", help="print liveness age + decision, then exit")
    parser.add_argument("--once", action="store_true", help="run a single evaluation, then exit")
    parser.add_argument("--dry-run", action="store_true", help="never actually reboot (log only)")
    parser.add_argument("--force-enabled", action="store_true",
                        help="ignore watchdog.enabled=false (for --check / testing)")
    args = parser.parse_args()

    wd, liveness_path, logs_dir, base = load_settings(args.config)
    logger = setup_logger("watchdog", str(logs_dir))

    if args.check:
        now = time.time()
        age = liveness_age_sec(liveness_path, now)
        watch = Watchdog(wd, liveness_path, base, logger, dry_run_override=True)
        watch.armed = (age is not None and age <= watch.stale_after)  # simulate "seen alive"
        decision = watch.evaluate(now)
        age_txt = "MISSING" if age is None else f"{age:.0f}s"
        print(json.dumps({
            "enabled": _as_bool(wd.get("enabled"), False),
            "liveness_file": str(liveness_path),
            "exists": liveness_path.exists(),
            "age": age_txt,
            "stale_after_sec": watch.stale_after,
            "recent_reboots_24h": len(watch.history.recent(now)),
            "decision": decision,
        }, ensure_ascii=False, indent=2))
        return 0

    if not _as_bool(wd.get("enabled"), False) and not args.force_enabled:
        logger.warning("watchdog.enabled=false — exiting (set watchdog.enabled: true in config to run)")
        return 0

    watch = Watchdog(wd, liveness_path, base, logger, dry_run_override=args.dry_run)
    watch.run(once=args.once)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
