"""
Performance metrics aggregation for controller dashboard.

This module reads:
- Daily event reports (reports/report_YYYY-MM-DD.csv)
- Role snapshot folders (barber/customer/wash/unknown)
- Runtime dashboard state (runtime/dashboard_state.json)

And produces compact statistics for UI rendering.
"""

from __future__ import annotations

import csv
import json
import re
from collections import defaultdict
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Dict, Optional


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
REPORT_FILE_RE = re.compile(r"^report_(\d{4}-\d{2}-\d{2})\.csv$")
GID_RE = re.compile(r"_G(\d+)\b", re.IGNORECASE)
SNAPSHOT_DAY_RE = re.compile(r"^(\d{8})_")
MANUAL_FEEDBACK_ORDER = ["haircut", "customerwash", "no_haircut", "no_customerwash"]


def _safe_rate(num: float, den: float) -> float:
    return float(num / den) if den > 0 else 0.0


def _to_day(v: Any) -> Optional[date]:
    if isinstance(v, date):
        return v
    if isinstance(v, datetime):
        return v.date()
    if isinstance(v, str):
        try:
            return datetime.strptime(v[:10], "%Y-%m-%d").date()
        except Exception:
            return None
    return None


def _parse_predicted_role_from_filename(path: Path) -> Optional[str]:
    name = path.name.upper()
    if "WASH_CUSTOMER" in name:
        return "wash_customer"
    if "_BARBER_" in name:
        return "barber"
    if "_CUSTOMER_" in name:
        return "customer"
    if "_UNKNOWN_" in name:
        return "unknown"
    return None


def _snapshot_capture_day(path: Path, fallback_day: date) -> date:
    m = SNAPSHOT_DAY_RE.match(path.name)
    if not m:
        return fallback_day
    try:
        return datetime.strptime(m.group(1), "%Y%m%d").date()
    except Exception:
        return fallback_day


class PerformanceMetrics:
    def __init__(
        self,
        project_root: Path,
        paths_cfg: Optional[Dict[str, Any]] = None,
        dashboard_state_file: Optional[Path] = None,
    ):
        self.project_root = Path(project_root)
        self.paths_cfg = dict(paths_cfg or {})
        self.reports_dir = self.project_root / str(self.paths_cfg.get("reports", "reports"))
        self.dashboard_state_file = (
            Path(dashboard_state_file)
            if dashboard_state_file is not None
            else (self.project_root / "runtime" / "dashboard_state.json")
        )

        staff_gallery = self.project_root / str(self.paths_cfg.get("staff_gallery", "data/staff_gallery"))
        self.role_dirs = {
            "barber": staff_gallery / "BARBER_UNIFORM",
            "customer": self.project_root / str(self.paths_cfg.get("customer_by_admin", "data/customer_by_admin")),
            "wash_customer": self.project_root
            / str(self.paths_cfg.get("customer_wash_by_admin", "data/customer_wash_by_admin")),
            "unknown": self.project_root / str(self.paths_cfg.get("unknown_by_admin", "data/unknown_by_admin")),
        }
        self.manual_feedback_root = self.project_root / str(
            self.paths_cfg.get("performance_feedback", "data/performance_feedback")
        )
        self.manual_feedback_dirs = {
            "haircut": self.manual_feedback_root / "haircut",
            "customerwash": self.manual_feedback_root / "customerwash",
            "no_haircut": self.manual_feedback_root / "no haircut",
            "no_customerwash": self.manual_feedback_root / "no customerwash",
        }
        self._ensure_manual_feedback_dirs()

    def collect(self, lookback_days: int = 14, as_of: Optional[date] = None) -> Dict[str, Any]:
        lookback_days = max(1, int(lookback_days))
        day_now = as_of or datetime.now().date()
        start_day = day_now - timedelta(days=lookback_days - 1)

        daily_stats = self._scan_reports(start_day=start_day, end_day=day_now)
        role_stats = self._scan_role_images(start_day=start_day, end_day=day_now, day_now=day_now)
        manual_feedback = self._scan_manual_feedback(start_day=start_day, end_day=day_now, day_now=day_now)
        runtime = self._read_runtime_state()

        by_date = {row["date"]: row for row in daily_stats}
        today_key = day_now.strftime("%Y-%m-%d")
        yesterday_key = (day_now - timedelta(days=1)).strftime("%Y-%m-%d")
        today = by_date.get(today_key, self._empty_daily(today_key))
        yesterday = by_date.get(yesterday_key, self._empty_daily(yesterday_key))

        haircut_7d = [d["haircuts"] for d in daily_stats[:7]]
        wash_7d = [d["washes"] for d in daily_stats[:7]]
        image_growth = role_stats.get("growth_by_day", [])
        manual_growth = manual_feedback.get("by_day", []) if isinstance(manual_feedback, dict) else []

        return {
            "as_of": today_key,
            "lookback_days": lookback_days,
            "runtime": runtime,
            "daily": daily_stats,
            "today": today,
            "chairs": self._flatten_chair_rows(daily_stats),
            "roles": role_stats,
            "manual_feedback": manual_feedback,
            "growth": {
                "events_today": int(today.get("total_events", 0)),
                "events_yesterday": int(yesterday.get("total_events", 0)),
                "events_delta": int(today.get("total_events", 0)) - int(yesterday.get("total_events", 0)),
                "haircut_7d_avg": round(sum(haircut_7d) / len(haircut_7d), 2) if haircut_7d else 0.0,
                "wash_7d_avg": round(sum(wash_7d) / len(wash_7d), 2) if wash_7d else 0.0,
                "images_today": int(role_stats.get("images_today", 0)),
                "images_7d": int(role_stats.get("images_7d", 0)),
                "image_by_day": image_growth[: min(len(image_growth), lookback_days)],
                "role_error_today_pct": float(role_stats.get("today_error_rate", 0.0)),
                "role_error_today_mismatch": int(role_stats.get("today_error_mismatch", 0)),
                "role_error_today_scorable": int(role_stats.get("today_error_scorable", 0)),
                "role_error_by_day": (role_stats.get("error_by_day", []) or [])[: min(lookback_days, 31)],
                "manual_error_today_pct": float(manual_feedback.get("today_error_rate", 0.0)),
                "manual_error_today_wrong": int(manual_feedback.get("today_wrong", 0)),
                "manual_error_today_labeled": int(manual_feedback.get("today_labeled", 0)),
                "manual_error_by_day": (manual_growth or [])[: min(lookback_days, 31)],
            },
        }

    def _ensure_manual_feedback_dirs(self):
        try:
            self.manual_feedback_root.mkdir(parents=True, exist_ok=True)
            for folder in self.manual_feedback_dirs.values():
                folder.mkdir(parents=True, exist_ok=True)
        except Exception:
            # Performance page should still work even when folder create fails.
            pass

    def _scan_reports(self, start_day: date, end_day: date) -> list[Dict[str, Any]]:
        out: list[Dict[str, Any]] = []
        if not self.reports_dir.exists():
            return out

        for report_file in sorted(self.reports_dir.glob("report_*.csv")):
            m = REPORT_FILE_RE.match(report_file.name)
            if not m:
                continue
            day_val = _to_day(m.group(1))
            if day_val is None or day_val < start_day or day_val > end_day:
                continue

            day_key = day_val.strftime("%Y-%m-%d")
            row = self._empty_daily(day_key)
            chair_map: Dict[tuple[str, str], Dict[str, Any]] = {}
            people_set = set()
            event_count = 0

            try:
                with open(report_file, "r", encoding="utf-8", newline="") as f:
                    reader = csv.DictReader(f)
                    for rec in reader:
                        event_count += 1
                        event_type = str(rec.get("event_type", "")).strip().lower()
                        camera = str(rec.get("camera", "")).strip()
                        zone = str(rec.get("zone_name", "")).strip()
                        person = str(rec.get("person_id", "")).strip()
                        if person:
                            people_set.add(person)

                        if event_type == "haircut":
                            row["haircuts"] += 1
                        elif event_type == "wash":
                            row["washes"] += 1
                        elif event_type == "wait":
                            row["waits"] += 1
                        elif event_type == "verified":
                            row["verified"] += 1

                        if zone.upper().startswith("CHAIR"):
                            key = (camera, zone)
                            if key not in chair_map:
                                chair_map[key] = {
                                    "camera": camera,
                                    "chair": zone,
                                    "haircuts": 0,
                                    "washes": 0,
                                    "event_count": 0,
                                    "people": set(),
                                }
                            item = chair_map[key]
                            if event_type == "haircut":
                                item["haircuts"] += 1
                            elif event_type == "wash":
                                item["washes"] += 1
                            item["event_count"] += 1
                            if person:
                                item["people"].add(person)
            except Exception:
                continue

            row["total_events"] = event_count
            row["unique_people"] = len(people_set)
            row["verified_rate"] = round(
                _safe_rate(row["verified"], (row["haircuts"] + row["washes"])), 4
            )

            chair_rows = []
            for item in chair_map.values():
                chair_rows.append(
                    {
                        "camera": item["camera"],
                        "chair": item["chair"],
                        "haircuts": int(item["haircuts"]),
                        "washes": int(item["washes"]),
                        "event_count": int(item["event_count"]),
                        "unique_people": len(item["people"]),
                    }
                )
            chair_rows.sort(key=lambda x: (x["camera"], x["chair"]))
            row["chairs"] = chair_rows
            out.append(row)

        out.sort(key=lambda x: x["date"], reverse=True)
        return out

    def _scan_role_images(self, start_day: date, end_day: date, day_now: date) -> Dict[str, Any]:
        stats_by_role: Dict[str, Dict[str, Any]] = {}
        growth_by_day = defaultdict(lambda: {"total": 0, "barber": 0, "customer": 0, "wash_customer": 0, "unknown": 0})
        error_by_day = defaultdict(lambda: {"scorable": 0, "match": 0, "mismatch": 0})

        total_images = 0
        images_today = 0
        images_7d = 0
        known_gid_union = set()
        known_images = 0
        unknown_images = 0
        proxy_match = 0
        proxy_mismatch = 0

        for role, folder in self.role_dirs.items():
            role_stat = {
                "images": 0,
                "unique_gids": 0,
                "added_today": 0,
                "added_7d": 0,
                "proxy_match": 0,
                "proxy_mismatch": 0,
                "proxy_unscorable": 0,
            }
            gids = set()

            if folder.exists():
                for p in folder.rglob("*"):
                    if not p.is_file() or p.suffix.lower() not in IMAGE_EXTS:
                        continue

                    role_stat["images"] += 1
                    total_images += 1
                    if role == "unknown":
                        unknown_images += 1
                    else:
                        known_images += 1

                    m_gid = GID_RE.search(p.name)
                    if m_gid:
                        gid_val = int(m_gid.group(1))
                        gids.add(gid_val)
                        if role != "unknown":
                            known_gid_union.add(gid_val)

                    mtime_day = datetime.fromtimestamp(p.stat().st_mtime).date()
                    capture_day = _snapshot_capture_day(p, mtime_day)
                    day_key = capture_day.strftime("%Y-%m-%d")
                    if start_day <= capture_day <= end_day:
                        growth_by_day[day_key]["total"] += 1
                        growth_by_day[day_key][role] += 1
                    if mtime_day == day_now:
                        role_stat["added_today"] += 1
                        images_today += 1
                    if (day_now - mtime_day).days <= 6:
                        role_stat["added_7d"] += 1
                        images_7d += 1

                    predicted = _parse_predicted_role_from_filename(p)
                    if predicted is None:
                        role_stat["proxy_unscorable"] += 1
                        continue
                    if predicted == role:
                        role_stat["proxy_match"] += 1
                        proxy_match += 1
                        if start_day <= capture_day <= end_day:
                            error_by_day[day_key]["scorable"] += 1
                            error_by_day[day_key]["match"] += 1
                    else:
                        role_stat["proxy_mismatch"] += 1
                        proxy_mismatch += 1
                        if start_day <= capture_day <= end_day:
                            error_by_day[day_key]["scorable"] += 1
                            error_by_day[day_key]["mismatch"] += 1

            role_stat["unique_gids"] = len(gids)
            scorable = role_stat["proxy_match"] + role_stat["proxy_mismatch"]
            role_stat["proxy_accuracy"] = round(_safe_rate(role_stat["proxy_match"], scorable), 4)
            stats_by_role[role] = role_stat

        scorable_total = proxy_match + proxy_mismatch
        growth_rows = []
        for day_key in sorted(growth_by_day.keys(), reverse=True):
            item = dict(growth_by_day[day_key])
            item["date"] = day_key
            growth_rows.append(item)

        error_rows = []
        for day_key in sorted(error_by_day.keys(), reverse=True):
            item = dict(error_by_day[day_key])
            scorable = int(item.get("scorable", 0))
            mismatch = int(item.get("mismatch", 0))
            item["date"] = day_key
            item["error_rate"] = round(_safe_rate(mismatch, scorable), 4)
            error_rows.append(item)

        today_key = day_now.strftime("%Y-%m-%d")
        today_item = error_by_day.get(today_key, {"scorable": 0, "match": 0, "mismatch": 0})
        today_scorable = int(today_item.get("scorable", 0))
        today_mismatch = int(today_item.get("mismatch", 0))

        return {
            "by_role": stats_by_role,
            "total_images": total_images,
            "known_images": known_images,
            "unknown_images": unknown_images,
            "unique_known_gids": len(known_gid_union),
            "images_today": images_today,
            "images_7d": images_7d,
            "classifiable_rate": round(_safe_rate(known_images, total_images), 4),
            "unknown_rate": round(_safe_rate(unknown_images, total_images), 4),
            "proxy_match": proxy_match,
            "proxy_mismatch": proxy_mismatch,
            "proxy_accuracy": round(_safe_rate(proxy_match, scorable_total), 4),
            "proxy_scorable": scorable_total,
            "growth_by_day": growth_rows,
            "error_by_day": error_rows,
            "today_error_rate": round(_safe_rate(today_mismatch, today_scorable), 4),
            "today_error_mismatch": today_mismatch,
            "today_error_scorable": today_scorable,
        }

    def _scan_manual_feedback(self, start_day: date, end_day: date, day_now: date) -> Dict[str, Any]:
        # Create folders on each scan to support first-run / deleted-folder recovery.
        self._ensure_manual_feedback_dirs()

        by_folder: Dict[str, Dict[str, Any]] = {}
        day_map = defaultdict(
            lambda: {
                "haircut": 0,
                "customerwash": 0,
                "no_haircut": 0,
                "no_customerwash": 0,
                "total": 0,
                "labeled": 0,
                "wrong": 0,
                "error_rate": 0.0,
            }
        )

        today_total = 0
        week_total = 0

        for key in MANUAL_FEEDBACK_ORDER:
            folder = self.manual_feedback_dirs.get(key)
            stat = {
                "folder": str(folder) if folder is not None else "",
                "images": 0,
                "added_today": 0,
                "added_7d": 0,
            }
            if folder is not None and folder.exists():
                for p in folder.rglob("*"):
                    if not p.is_file() or p.suffix.lower() not in IMAGE_EXTS:
                        continue
                    stat["images"] += 1
                    mtime_day = datetime.fromtimestamp(p.stat().st_mtime).date()
                    if mtime_day == day_now:
                        stat["added_today"] += 1
                        today_total += 1
                    if (day_now - mtime_day).days <= 6:
                        stat["added_7d"] += 1
                        week_total += 1
                    if start_day <= mtime_day <= end_day:
                        day_key = mtime_day.strftime("%Y-%m-%d")
                        day_map[day_key][key] += 1
                        day_map[day_key]["total"] += 1
            by_folder[key] = stat

        for day_key, item in day_map.items():
            labeled = int(item.get("haircut", 0)) + int(item.get("customerwash", 0)) + int(item.get("no_haircut", 0)) + int(item.get("no_customerwash", 0))
            wrong = int(item.get("no_haircut", 0)) + int(item.get("no_customerwash", 0))
            item["labeled"] = labeled
            item["wrong"] = wrong
            item["error_rate"] = round(_safe_rate(wrong, labeled), 4)

        by_day = []
        for day_key in sorted(day_map.keys(), reverse=True):
            row = dict(day_map[day_key])
            row["date"] = day_key
            by_day.append(row)

        haircut_correct = int(by_folder.get("haircut", {}).get("images", 0))
        haircut_wrong = int(by_folder.get("no_haircut", {}).get("images", 0))
        customerwash_correct = int(by_folder.get("customerwash", {}).get("images", 0))
        customerwash_wrong = int(by_folder.get("no_customerwash", {}).get("images", 0))

        today_key = day_now.strftime("%Y-%m-%d")
        today_item = day_map.get(today_key, {})
        today_labeled = int(today_item.get("labeled", 0))
        today_wrong = int(today_item.get("wrong", 0))

        labeled_total = haircut_correct + haircut_wrong + customerwash_correct + customerwash_wrong
        wrong_total = haircut_wrong + customerwash_wrong

        return {
            "root": str(self.manual_feedback_root),
            "by_folder": by_folder,
            "total_images": int(sum(int(v.get("images", 0)) for v in by_folder.values())),
            "today_total": int(today_total),
            "week_total": int(week_total),
            "haircut_correct": haircut_correct,
            "haircut_wrong": haircut_wrong,
            "haircut_precision": round(_safe_rate(haircut_correct, haircut_correct + haircut_wrong), 4),
            "customerwash_correct": customerwash_correct,
            "customerwash_wrong": customerwash_wrong,
            "customerwash_precision": round(_safe_rate(customerwash_correct, customerwash_correct + customerwash_wrong), 4),
            "overall_labeled": int(labeled_total),
            "overall_wrong": int(wrong_total),
            "overall_error_rate": round(_safe_rate(wrong_total, labeled_total), 4),
            "today_labeled": today_labeled,
            "today_wrong": today_wrong,
            "today_error_rate": round(_safe_rate(today_wrong, today_labeled), 4),
            "by_day": by_day,
        }

    def _read_runtime_state(self) -> Dict[str, Any]:
        if not self.dashboard_state_file.exists():
            return {}
        try:
            with open(self.dashboard_state_file, "r", encoding="utf-8") as f:
                payload = json.load(f)
        except Exception:
            return {}

        status = payload.get("status", {}) if isinstance(payload, dict) else {}
        summary = payload.get("summary", {}) if isinstance(payload, dict) else {}
        rt_counts = status.get("realtime_counts", {}) if isinstance(status, dict) else {}
        return {
            "running": bool(status.get("running", False)),
            "active_tracks": int(status.get("active_tracks", summary.get("active_people", 0) or 0)),
            "chairs_total_rt": int(rt_counts.get("chairs_total", 0) or 0),
            "washes_total_rt": int(rt_counts.get("washes_total", 0) or 0),
            "waits_total_rt": int(rt_counts.get("waits_total", 0) or 0),
            "haircuts_total": int(summary.get("haircuts", 0) or 0),
            "washes_total": int(summary.get("washes", 0) or 0),
            "verified_total": int(summary.get("verified", 0) or 0),
            "role_unknown_rate_rt": float(summary.get("role_unknown_rate", 0.0) or 0.0),
            "fsm_open_sessions": int(summary.get("fsm_open_sessions", 0) or 0),
        }

    @staticmethod
    def _empty_daily(day_key: str) -> Dict[str, Any]:
        return {
            "date": day_key,
            "haircuts": 0,
            "washes": 0,
            "waits": 0,
            "verified": 0,
            "verified_rate": 0.0,
            "total_events": 0,
            "unique_people": 0,
            "chairs": [],
        }

    @staticmethod
    def _flatten_chair_rows(daily_stats: list[Dict[str, Any]]) -> list[Dict[str, Any]]:
        rows = []
        for d in daily_stats:
            day_key = str(d.get("date", ""))
            for item in d.get("chairs", []) or []:
                rows.append(
                    {
                        "date": day_key,
                        "camera": str(item.get("camera", "")),
                        "chair": str(item.get("chair", "")),
                        "haircuts": int(item.get("haircuts", 0)),
                        "washes": int(item.get("washes", 0)),
                        "event_count": int(item.get("event_count", 0)),
                        "unique_people": int(item.get("unique_people", 0)),
                    }
                )
        rows.sort(key=lambda x: (x["date"], x["camera"], x["chair"]), reverse=True)
        return rows
