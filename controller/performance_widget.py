"""
Performance analytics widget for HGCameraCounter.

Features:
- Day-to-day trend lines by class
- Timeline with event points + time
- Snapshot preview for selected event
- Daily / Monthly / Yearly summaries
- Adjustable date range presets and custom range
"""

from __future__ import annotations

import csv
import json
import re
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

from PySide6.QtCharts import (
    QCategoryAxis,
    QChart,
    QChartView,
    QDateTimeAxis,
    QLineSeries,
    QScatterSeries,
    QValueAxis,
)
from PySide6.QtCore import QDate, QDateTime, Qt, QTime, QUrl
from PySide6.QtGui import QDesktopServices, QFont, QPainter, QPixmap
from PySide6.QtWidgets import (
    QComboBox,
    QDateEdit,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)


REPORT_RE = re.compile(r"^report_(\d{4}-\d{2}-\d{2})\.csv$")
QUALITY_TS_RE = re.compile(r"^(\d{8})_(\d{6})")
CLASS_ORDER = ["haircut", "wash", "wait", "verified"]
CLASS_COLORS = {
    "haircut": Qt.green,
    "wash": Qt.blue,
    "wait": Qt.darkYellow,
    "verified": Qt.magenta,
}
QUALITY_CLASSES = ["haircut", "wash", "staff"]


@dataclass
class EventItem:
    timestamp: datetime
    event_type: str
    camera: str
    zone_name: str
    person_id: str
    snapshot_path: Optional[Path]
    feedback_snapshot_path: Optional[Path]
    metadata: Dict[str, Any]

    @property
    def date_key(self) -> str:
        return self.timestamp.strftime("%Y-%m-%d")

    @property
    def month_key(self) -> str:
        return self.timestamp.strftime("%Y-%m")

    @property
    def year_key(self) -> str:
        return self.timestamp.strftime("%Y")


@dataclass
class QualitySample:
    timestamp: datetime
    true_class: str
    predicted_class: str
    source_path: Path

    @property
    def day_key(self) -> str:
        return self.timestamp.strftime("%Y-%m-%d")

    @property
    def month_key(self) -> str:
        return self.timestamp.strftime("%Y-%m")

    @property
    def year_key(self) -> str:
        return self.timestamp.strftime("%Y")


class PerformanceWidget(QWidget):
    def __init__(self, config: Dict[str, Any], parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.config = config or {}
        self.project_root = Path(__file__).resolve().parent.parent
        paths_cfg = self.config.get("paths", {}) or {}
        self.reports_dir = self.project_root / str(paths_cfg.get("reports", "reports"))
        self.events: List[EventItem] = []
        self.events_filtered: List[EventItem] = []
        self.current_timeline_events: List[EventItem] = []
        self.quality_samples: List[QualitySample] = []
        self._timeline_class_to_index = {name: idx for idx, name in enumerate(CLASS_ORDER)}
        feedback_root = self.project_root / str(paths_cfg.get("performance_feedback", "data/performance_feedback"))
        self.quality_true_dirs: Dict[str, List[Path]] = {
            "haircut": [
                feedback_root / "haircut",
                self.project_root / str(paths_cfg.get("customer_by_admin", "data/customer_by_admin")),
            ],
            "wash": [
                feedback_root / "customerwash",
                self.project_root / str(paths_cfg.get("customer_wash_by_admin", "data/customer_wash_by_admin")),
            ],
            "staff": [
                (self.project_root / str(paths_cfg.get("staff_gallery", "data/staff_gallery")) / "BARBER_UNIFORM"),
            ],
        }

        self._build_ui()
        self._init_default_dates()
        self.refresh_data()

    def _build_ui(self) -> None:
        outer = QVBoxLayout(self)
        self.scroll_area = QScrollArea(self)
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        outer.addWidget(self.scroll_area)

        content = QWidget()
        self.scroll_area.setWidget(content)
        root = QVBoxLayout(content)

        title = QLabel("Performance Analytics")
        title.setFont(QFont("Arial", 13, QFont.Bold))
        root.addWidget(title)

        control_box = QGroupBox("Filters")
        control_form = QGridLayout()
        self.range_preset_combo = QComboBox()
        self.range_preset_combo.addItems(["7 days", "30 days", "90 days", "365 days", "All", "Custom"])
        self.range_preset_combo.currentTextChanged.connect(self._on_preset_changed)
        control_form.addWidget(QLabel("Range Preset"), 0, 0)
        control_form.addWidget(self.range_preset_combo, 0, 1)

        self.start_date_edit = QDateEdit()
        self.start_date_edit.setCalendarPopup(True)
        self.start_date_edit.dateChanged.connect(self._on_custom_date_changed)
        control_form.addWidget(QLabel("Start Date"), 0, 2)
        control_form.addWidget(self.start_date_edit, 0, 3)

        self.end_date_edit = QDateEdit()
        self.end_date_edit.setCalendarPopup(True)
        self.end_date_edit.dateChanged.connect(self._on_custom_date_changed)
        control_form.addWidget(QLabel("End Date"), 0, 4)
        control_form.addWidget(self.end_date_edit, 0, 5)

        self.class_filter_combo = QComboBox()
        self.class_filter_combo.addItems(["all"] + CLASS_ORDER)
        self.class_filter_combo.currentTextChanged.connect(self._rebuild_views_only)
        control_form.addWidget(QLabel("Timeline Class Filter"), 1, 0)
        control_form.addWidget(self.class_filter_combo, 1, 1)

        self.timeline_date_edit = QDateEdit()
        self.timeline_date_edit.setCalendarPopup(True)
        self.timeline_date_edit.dateChanged.connect(self._rebuild_views_only)
        control_form.addWidget(QLabel("Timeline Date"), 1, 2)
        control_form.addWidget(self.timeline_date_edit, 1, 3)

        self.refresh_btn = QPushButton("Refresh")
        self.refresh_btn.clicked.connect(self.refresh_data)
        control_form.addWidget(self.refresh_btn, 1, 4)

        self.open_snapshot_btn = QPushButton("Open Selected Snapshot")
        self.open_snapshot_btn.clicked.connect(self.open_selected_snapshot)
        control_form.addWidget(self.open_snapshot_btn, 1, 5)
        control_box.setLayout(control_form)
        root.addWidget(control_box)

        summary_box = QGroupBox("Summary")
        summary_layout = QGridLayout()
        self.summary_labels: Dict[str, QLabel] = {}
        summary_keys = [
            "selected_total",
            "selected_haircut",
            "selected_wash",
            "selected_wait",
            "selected_verified",
            "selected_unique_people",
            "today_total",
            "month_total",
            "year_total",
        ]
        for idx, key in enumerate(summary_keys):
            label_name = key.replace("_", " ").title()
            summary_layout.addWidget(QLabel(label_name), idx // 3, (idx % 3) * 2)
            lbl = QLabel("0")
            lbl.setStyleSheet("font-weight: bold; color: #0D47A1;")
            self.summary_labels[key] = lbl
            summary_layout.addWidget(lbl, idx // 3, (idx % 3) * 2 + 1)
        summary_box.setLayout(summary_layout)
        root.addWidget(summary_box)

        quality_box = QGroupBox("AI Intelligence Measurement (Confusion Matrix / Precision / Recall)")
        quality_layout = QVBoxLayout(quality_box)

        quality_ctrl = QHBoxLayout()
        quality_ctrl.addWidget(QLabel("Quality Trend"))
        self.quality_granularity_combo = QComboBox()
        self.quality_granularity_combo.addItems(["Day", "Month", "Year"])
        self.quality_granularity_combo.currentTextChanged.connect(self._rebuild_quality_views_only)
        quality_ctrl.addWidget(self.quality_granularity_combo)
        quality_ctrl.addStretch()
        quality_layout.addLayout(quality_ctrl)

        quality_split = QSplitter(Qt.Horizontal)
        self.confusion_table = QTableWidget(0, 5)
        self.confusion_table.setHorizontalHeaderLabels(
            ["True Class", "Pred Haircut", "Pred Wash", "Pred Staff", "Pred Unknown"]
        )
        self.confusion_table.horizontalHeader().setStretchLastSection(True)
        quality_split.addWidget(self._wrap_group("Confusion Matrix", self.confusion_table))

        self.prf_table = QTableWidget(0, 6)
        self.prf_table.setHorizontalHeaderLabels(["Class", "Precision", "Recall", "F1", "Support", "Unknown Pred"])
        self.prf_table.horizontalHeader().setStretchLastSection(True)
        quality_split.addWidget(self._wrap_group("Precision / Recall / F1", self.prf_table))
        quality_split.setStretchFactor(0, 1)
        quality_split.setStretchFactor(1, 1)
        quality_layout.addWidget(quality_split)

        self.quality_chart = QChart()
        self.quality_chart.setTitle("Intelligence Trend (Macro Precision/Recall/F1)")
        self.quality_chart_view = QChartView(self.quality_chart)
        self.quality_chart_view.setRenderHint(QPainter.Antialiasing)
        self.quality_chart_view.setMinimumHeight(220)
        quality_layout.addWidget(self.quality_chart_view)
        root.addWidget(quality_box)

        self.daily_chart = QChart()
        self.daily_chart.setTitle("Day-to-Day Detection by Class")
        self.daily_chart_view = QChartView(self.daily_chart)
        self.daily_chart_view.setRenderHint(QPainter.Antialiasing)
        self.daily_chart_view.setMinimumHeight(260)

        self.timeline_chart = QChart()
        self.timeline_chart.setTitle("Timeline Events (selected day)")
        self.timeline_chart_view = QChartView(self.timeline_chart)
        self.timeline_chart_view.setRenderHint(QPainter.Antialiasing)
        self.timeline_chart_view.setMinimumHeight(260)

        chart_split = QSplitter(Qt.Horizontal)
        left_chart = QWidget()
        left_chart_layout = QVBoxLayout(left_chart)
        left_chart_layout.addWidget(self.daily_chart_view)
        right_chart = QWidget()
        right_chart_layout = QVBoxLayout(right_chart)
        right_chart_layout.addWidget(self.timeline_chart_view)
        chart_split.addWidget(left_chart)
        chart_split.addWidget(right_chart)
        chart_split.setStretchFactor(0, 1)
        chart_split.setStretchFactor(1, 1)
        root.addWidget(chart_split)

        lower_split = QSplitter(Qt.Horizontal)
        self.events_table = QTableWidget(0, 7)
        self.events_table.setHorizontalHeaderLabels(
            ["Time", "Class", "Camera", "Zone", "Person", "Snapshot", "Feedback Snapshot"]
        )
        self.events_table.horizontalHeader().setStretchLastSection(True)
        self.events_table.cellClicked.connect(self._on_event_row_selected)
        self.events_table.cellDoubleClicked.connect(lambda *_: self.open_selected_snapshot())
        lower_split.addWidget(self.events_table)

        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        self.snapshot_preview = QLabel("No snapshot selected")
        self.snapshot_preview.setAlignment(Qt.AlignCenter)
        self.snapshot_preview.setMinimumHeight(240)
        self.snapshot_preview.setStyleSheet("border: 1px solid #BDBDBD; background: #FAFAFA;")
        right_layout.addWidget(self.snapshot_preview)
        self.event_detail_text = QTextEdit()
        self.event_detail_text.setReadOnly(True)
        right_layout.addWidget(self.event_detail_text)
        lower_split.addWidget(right_panel)
        lower_split.setStretchFactor(0, 3)
        lower_split.setStretchFactor(1, 2)
        root.addWidget(lower_split)

        summary_split = QSplitter(Qt.Horizontal)
        self.daily_summary_table = self._make_summary_table(["Date", "Total", "Haircut", "Wash", "Wait", "Verified", "People"])
        self.monthly_summary_table = self._make_summary_table(["Month", "Total", "Haircut", "Wash", "Wait", "Verified", "People"])
        self.yearly_summary_table = self._make_summary_table(["Year", "Total", "Haircut", "Wash", "Wait", "Verified", "People"])
        summary_split.addWidget(self._wrap_group("Daily Summary", self.daily_summary_table))
        summary_split.addWidget(self._wrap_group("Monthly Summary", self.monthly_summary_table))
        summary_split.addWidget(self._wrap_group("Yearly Summary", self.yearly_summary_table))
        summary_split.setStretchFactor(0, 1)
        summary_split.setStretchFactor(1, 1)
        summary_split.setStretchFactor(2, 1)
        root.addWidget(summary_split)

    @staticmethod
    def _make_summary_table(headers: List[str]) -> QTableWidget:
        tb = QTableWidget(0, len(headers))
        tb.setHorizontalHeaderLabels(headers)
        tb.horizontalHeader().setStretchLastSection(True)
        return tb

    @staticmethod
    def _wrap_group(title: str, child: QWidget) -> QWidget:
        box = QGroupBox(title)
        lay = QVBoxLayout(box)
        lay.addWidget(child)
        return box

    def _init_default_dates(self) -> None:
        min_day, max_day = self._report_min_max_dates()
        today = QDate.currentDate()
        end_day = max_day if max_day is not None else today
        start_day = end_day.addDays(-29)
        if min_day is not None and start_day < min_day:
            start_day = min_day
        self.end_date_edit.setDate(end_day)
        self.start_date_edit.setDate(start_day)
        self.timeline_date_edit.setDate(end_day)
        self.range_preset_combo.setCurrentText("30 days")

    def _report_min_max_dates(self) -> tuple[Optional[QDate], Optional[QDate]]:
        if not self.reports_dir.exists():
            return None, None
        days: List[QDate] = []
        for csv_path in self.reports_dir.glob("report_*.csv"):
            m = REPORT_RE.match(csv_path.name)
            if not m:
                continue
            try:
                d = datetime.strptime(m.group(1), "%Y-%m-%d")
                days.append(QDate(d.year, d.month, d.day))
            except Exception:
                continue
        if not days:
            return None, None
        days.sort()
        return days[0], days[-1]

    def _on_preset_changed(self, text: str) -> None:
        if text == "Custom":
            return
        today = QDate.currentDate()
        mapping = {
            "7 days": 6,
            "30 days": 29,
            "90 days": 89,
            "365 days": 364,
        }
        min_day, max_day = self._report_min_max_dates()
        if text == "All":
            if min_day is None or max_day is None:
                return
            self.end_date_edit.blockSignals(True)
            self.start_date_edit.blockSignals(True)
            self.end_date_edit.setDate(max_day)
            self.start_date_edit.setDate(min_day)
            self.end_date_edit.blockSignals(False)
            self.start_date_edit.blockSignals(False)
            self.timeline_date_edit.setDate(max_day)
            self.refresh_data()
            return
        days_back = mapping.get(text, 29)
        self.end_date_edit.blockSignals(True)
        self.start_date_edit.blockSignals(True)
        self.end_date_edit.setDate(today)
        self.start_date_edit.setDate(today.addDays(-days_back))
        self.end_date_edit.blockSignals(False)
        self.start_date_edit.blockSignals(False)
        self.timeline_date_edit.setDate(self.end_date_edit.date())
        self.refresh_data()

    def _on_custom_date_changed(self, _date: QDate) -> None:
        if self.range_preset_combo.currentText() != "Custom":
            self.range_preset_combo.blockSignals(True)
            self.range_preset_combo.setCurrentText("Custom")
            self.range_preset_combo.blockSignals(False)
        self.refresh_data()

    def _range_dates(self) -> tuple[QDate, QDate]:
        start = self.start_date_edit.date()
        end = self.end_date_edit.date()
        if start > end:
            start, end = end, start
        return start, end

    def refresh_data(self) -> None:
        start_qd, end_qd = self._range_dates()
        tday = self.timeline_date_edit.date()
        if tday < start_qd or tday > end_qd:
            self.timeline_date_edit.blockSignals(True)
            self.timeline_date_edit.setDate(end_qd)
            self.timeline_date_edit.blockSignals(False)
        self.events = self._load_events(start_qd, end_qd)
        self.events_filtered = self._class_filtered_events(self.events)
        self._update_summary_cards(self.events)
        self._build_daily_chart(self.events)
        self._build_timeline_chart(self.events_filtered)
        self._fill_events_table(self.events_filtered)
        self._fill_summary_tables(self.events)
        self.quality_samples = self._load_quality_samples(start_qd, end_qd)
        self._fill_quality_tables(self.quality_samples)
        self._build_quality_trend_chart(self.quality_samples)

    def _rebuild_views_only(self) -> None:
        self.events_filtered = self._class_filtered_events(self.events)
        self._build_timeline_chart(self.events_filtered)
        self._fill_events_table(self.events_filtered)

    def _rebuild_quality_views_only(self) -> None:
        self._fill_quality_tables(self.quality_samples)
        self._build_quality_trend_chart(self.quality_samples)

    def _class_filtered_events(self, events: List[EventItem]) -> List[EventItem]:
        selected = self.class_filter_combo.currentText().strip().lower()
        if selected == "all":
            return list(events)
        return [e for e in events if e.event_type == selected]

    def _load_events(self, start_qd: QDate, end_qd: QDate) -> List[EventItem]:
        start_day = datetime(start_qd.year(), start_qd.month(), start_qd.day())
        end_day = datetime(end_qd.year(), end_qd.month(), end_qd.day()) + timedelta(days=1) - timedelta(milliseconds=1)
        out: List[EventItem] = []

        if not self.reports_dir.exists():
            return out

        for csv_path in sorted(self.reports_dir.glob("report_*.csv")):
            m = REPORT_RE.match(csv_path.name)
            if not m:
                continue
            try:
                file_day = datetime.strptime(m.group(1), "%Y-%m-%d")
            except Exception:
                continue
            if file_day.date() < start_day.date() or file_day.date() > end_day.date():
                continue

            try:
                with open(csv_path, "r", encoding="utf-8", newline="") as f:
                    reader = csv.DictReader(f)
                    for rec in reader:
                        ts_raw = str(rec.get("timestamp", "")).strip()
                        try:
                            ts = datetime.fromisoformat(ts_raw)
                        except Exception:
                            continue
                        if ts < start_day or ts > end_day:
                            continue
                        event_type = str(rec.get("event_type", "")).strip().lower()
                        if event_type not in CLASS_ORDER:
                            continue
                        metadata_raw = rec.get("metadata", "")
                        metadata = {}
                        if metadata_raw:
                            try:
                                metadata = json.loads(metadata_raw)
                            except Exception:
                                metadata = {}

                        snap = self._resolve_snapshot_path(metadata.get("snapshot_path"))
                        fb_snap = self._resolve_snapshot_path(metadata.get("feedback_snapshot_path"))
                        out.append(
                            EventItem(
                                timestamp=ts,
                                event_type=event_type,
                                camera=str(rec.get("camera", "")),
                                zone_name=str(rec.get("zone_name", "")),
                                person_id=str(rec.get("person_id", "")),
                                snapshot_path=snap,
                                feedback_snapshot_path=fb_snap,
                                metadata=metadata if isinstance(metadata, dict) else {},
                            )
                        )
            except Exception:
                continue

        out.sort(key=lambda x: x.timestamp)
        return out

    @staticmethod
    def _parse_capture_dt_from_filename(path: Path) -> Optional[datetime]:
        m = QUALITY_TS_RE.match(path.name)
        if not m:
            return None
        try:
            return datetime.strptime(f"{m.group(1)}_{m.group(2)}", "%Y%m%d_%H%M%S")
        except Exception:
            return None

    @staticmethod
    def _predict_quality_class_from_filename(path: Path) -> str:
        name = path.name.upper()
        if any(tok in name for tok in ("BARBER", "STAFF")):
            return "staff"
        if any(tok in name for tok in ("WASH_CUSTOMER", "CUSTOMERWASH", "CUSTOMER_WASH", "SHAMPOO", "WASH")):
            return "wash"
        if any(tok in name for tok in ("HAIRCUT", "CHAIR", "CUSTOMER")):
            return "haircut"
        return "unknown"

    def _load_quality_samples(self, start_qd: QDate, end_qd: QDate) -> List[QualitySample]:
        start_day = datetime(start_qd.year(), start_qd.month(), start_qd.day())
        end_day = datetime(end_qd.year(), end_qd.month(), end_qd.day()) + timedelta(days=1) - timedelta(milliseconds=1)
        out: List[QualitySample] = []
        seen_paths = set()
        for true_class, folders in self.quality_true_dirs.items():
            for folder in folders:
                if not folder.exists():
                    continue
                for p in folder.rglob("*"):
                    if (not p.is_file()) or (p.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}):
                        continue
                    rp = str(p.resolve())
                    if rp in seen_paths:
                        continue
                    seen_paths.add(rp)
                    ts = self._parse_capture_dt_from_filename(p)
                    if ts is None:
                        ts = datetime.fromtimestamp(p.stat().st_mtime)
                    if ts < start_day or ts > end_day:
                        continue
                    pred = self._predict_quality_class_from_filename(p)
                    out.append(
                        QualitySample(
                            timestamp=ts,
                            true_class=true_class,
                            predicted_class=pred,
                            source_path=p,
                        )
                    )
        out.sort(key=lambda x: x.timestamp)
        return out

    def _compute_quality_stats(self, samples: List[QualitySample]) -> Dict[str, Any]:
        matrix: Dict[str, Dict[str, int]] = {
            true_c: {pred_c: 0 for pred_c in QUALITY_CLASSES}
            for true_c in QUALITY_CLASSES
        }
        unknown_pred: Dict[str, int] = {c: 0 for c in QUALITY_CLASSES}
        for s in samples:
            t = s.true_class if s.true_class in QUALITY_CLASSES else None
            if t is None:
                continue
            p = s.predicted_class if s.predicted_class in QUALITY_CLASSES else "unknown"
            if p == "unknown":
                unknown_pred[t] += 1
            else:
                matrix[t][p] += 1

        per_class: Dict[str, Dict[str, float]] = {}
        for cls_name in QUALITY_CLASSES:
            tp = float(matrix[cls_name][cls_name])
            fp = float(sum(matrix[t][cls_name] for t in QUALITY_CLASSES if t != cls_name))
            fn = float(sum(matrix[cls_name][p] for p in QUALITY_CLASSES if p != cls_name) + unknown_pred[cls_name])
            support = float(sum(matrix[cls_name].values()) + unknown_pred[cls_name])
            precision = (tp / (tp + fp)) if (tp + fp) > 0 else 0.0
            recall = (tp / (tp + fn)) if (tp + fn) > 0 else 0.0
            f1 = (2.0 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
            per_class[cls_name] = {
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "support": support,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "unknown_pred": float(unknown_pred[cls_name]),
            }

        macro_precision = sum(per_class[c]["precision"] for c in QUALITY_CLASSES) / float(len(QUALITY_CLASSES))
        macro_recall = sum(per_class[c]["recall"] for c in QUALITY_CLASSES) / float(len(QUALITY_CLASSES))
        macro_f1 = sum(per_class[c]["f1"] for c in QUALITY_CLASSES) / float(len(QUALITY_CLASSES))
        return {
            "matrix": matrix,
            "unknown_pred": unknown_pred,
            "per_class": per_class,
            "macro_precision": macro_precision,
            "macro_recall": macro_recall,
            "macro_f1": macro_f1,
            "samples": len(samples),
        }

    def _fill_quality_tables(self, samples: List[QualitySample]) -> None:
        stats = self._compute_quality_stats(samples)
        matrix = stats.get("matrix", {})
        unknown_map = stats.get("unknown_pred", {})

        self.confusion_table.setRowCount(0)
        for true_cls in QUALITY_CLASSES:
            row = self.confusion_table.rowCount()
            self.confusion_table.insertRow(row)
            self.confusion_table.setItem(row, 0, QTableWidgetItem(true_cls))
            self.confusion_table.setItem(row, 1, QTableWidgetItem(str(int(matrix.get(true_cls, {}).get("haircut", 0)))))
            self.confusion_table.setItem(row, 2, QTableWidgetItem(str(int(matrix.get(true_cls, {}).get("wash", 0)))))
            self.confusion_table.setItem(row, 3, QTableWidgetItem(str(int(matrix.get(true_cls, {}).get("staff", 0)))))
            self.confusion_table.setItem(row, 4, QTableWidgetItem(str(int(unknown_map.get(true_cls, 0)))))

        self.prf_table.setRowCount(0)
        per_class = stats.get("per_class", {})
        for cls_name in QUALITY_CLASSES:
            item = per_class.get(cls_name, {})
            row = self.prf_table.rowCount()
            self.prf_table.insertRow(row)
            self.prf_table.setItem(row, 0, QTableWidgetItem(cls_name))
            self.prf_table.setItem(row, 1, QTableWidgetItem(f"{float(item.get('precision', 0.0)) * 100.0:.2f}%"))
            self.prf_table.setItem(row, 2, QTableWidgetItem(f"{float(item.get('recall', 0.0)) * 100.0:.2f}%"))
            self.prf_table.setItem(row, 3, QTableWidgetItem(f"{float(item.get('f1', 0.0)) * 100.0:.2f}%"))
            self.prf_table.setItem(row, 4, QTableWidgetItem(str(int(item.get("support", 0)))))
            self.prf_table.setItem(row, 5, QTableWidgetItem(str(int(item.get("unknown_pred", 0)))))

        macro_row = self.prf_table.rowCount()
        self.prf_table.insertRow(macro_row)
        self.prf_table.setItem(macro_row, 0, QTableWidgetItem("macro_avg"))
        self.prf_table.setItem(macro_row, 1, QTableWidgetItem(f"{float(stats.get('macro_precision', 0.0)) * 100.0:.2f}%"))
        self.prf_table.setItem(macro_row, 2, QTableWidgetItem(f"{float(stats.get('macro_recall', 0.0)) * 100.0:.2f}%"))
        self.prf_table.setItem(macro_row, 3, QTableWidgetItem(f"{float(stats.get('macro_f1', 0.0)) * 100.0:.2f}%"))
        self.prf_table.setItem(macro_row, 4, QTableWidgetItem(str(int(stats.get("samples", 0)))))
        self.prf_table.setItem(macro_row, 5, QTableWidgetItem("-"))

    def _quality_bucket_key(self, sample: QualitySample) -> str:
        mode = self.quality_granularity_combo.currentText().strip().lower()
        if mode == "month":
            return sample.month_key
        if mode == "year":
            return sample.year_key
        return sample.day_key

    def _build_quality_trend_chart(self, samples: List[QualitySample]) -> None:
        self.quality_chart.removeAllSeries()
        for ax in self.quality_chart.axes():
            self.quality_chart.removeAxis(ax)

        groups: Dict[str, List[QualitySample]] = defaultdict(list)
        for s in samples:
            groups[self._quality_bucket_key(s)].append(s)
        keys = sorted(groups.keys())
        if not keys:
            self.quality_chart.setTitle("Intelligence Trend (Macro Precision/Recall/F1) - no data")
            return

        x_axis = QCategoryAxis()
        x_axis.setTitleText(self.quality_granularity_combo.currentText())
        x_axis.setStartValue(0.0)
        x_axis.setRange(0.0, float(max(len(keys) - 1, 1)))
        for i, key in enumerate(keys):
            x_axis.append(key, float(i))

        y_axis = QValueAxis()
        y_axis.setTitleText("Score (%)")
        y_axis.setRange(0.0, 100.0)
        y_axis.setTickCount(6)

        metric_defs = [
            ("Macro Precision", "macro_precision", Qt.darkGreen),
            ("Macro Recall", "macro_recall", Qt.darkBlue),
            ("Macro F1", "macro_f1", Qt.darkRed),
        ]
        for series_name, metric_key, color in metric_defs:
            series = QLineSeries()
            series.setName(series_name)
            pen = series.pen()
            pen.setColor(color)
            pen.setWidth(2)
            series.setPen(pen)
            for i, key in enumerate(keys):
                stat = self._compute_quality_stats(groups[key])
                series.append(float(i), float(stat.get(metric_key, 0.0)) * 100.0)
            self.quality_chart.addSeries(series)

        self.quality_chart.addAxis(x_axis, Qt.AlignBottom)
        self.quality_chart.addAxis(y_axis, Qt.AlignLeft)
        for s in self.quality_chart.series():
            s.attachAxis(x_axis)
            s.attachAxis(y_axis)
        self.quality_chart.legend().setVisible(True)
        self.quality_chart.setTitle(
            f"Intelligence Trend ({self.quality_granularity_combo.currentText()} - Macro Precision/Recall/F1)"
        )

    def _resolve_snapshot_path(self, raw: Any) -> Optional[Path]:
        if not raw:
            return None
        txt = str(raw).strip().replace("\\", "/")
        p = Path(txt)
        if p.is_absolute():
            return p if p.exists() else None
        direct = (self.project_root / p).resolve()
        if direct.exists():
            return direct
        alt_data = (self.project_root / "data" / p).resolve()
        if alt_data.exists():
            return alt_data
        return None

    def _update_summary_cards(self, events: List[EventItem]) -> None:
        counts = {k: 0 for k in CLASS_ORDER}
        persons = set()
        for e in events:
            counts[e.event_type] = counts.get(e.event_type, 0) + 1
            if e.person_id:
                persons.add(e.person_id)

        self.summary_labels["selected_total"].setText(str(len(events)))
        self.summary_labels["selected_haircut"].setText(str(counts.get("haircut", 0)))
        self.summary_labels["selected_wash"].setText(str(counts.get("wash", 0)))
        self.summary_labels["selected_wait"].setText(str(counts.get("wait", 0)))
        self.summary_labels["selected_verified"].setText(str(counts.get("verified", 0)))
        self.summary_labels["selected_unique_people"].setText(str(len(persons)))

        today = QDate.currentDate().toPython()
        month_key = today.strftime("%Y-%m")
        year_key = today.strftime("%Y")
        today_total = 0
        month_total = 0
        year_total = 0
        for e in events:
            if e.timestamp.date() == today:
                today_total += 1
            if e.month_key == month_key:
                month_total += 1
            if e.year_key == year_key:
                year_total += 1
        self.summary_labels["today_total"].setText(str(today_total))
        self.summary_labels["month_total"].setText(str(month_total))
        self.summary_labels["year_total"].setText(str(year_total))

    def _build_daily_chart(self, events: List[EventItem]) -> None:
        self.daily_chart.removeAllSeries()
        for ax in self.daily_chart.axes():
            self.daily_chart.removeAxis(ax)

        by_day: Dict[str, Dict[str, int]] = {}
        for e in events:
            if e.date_key not in by_day:
                by_day[e.date_key] = {k: 0 for k in CLASS_ORDER}
            by_day[e.date_key][e.event_type] += 1
        day_keys = sorted(by_day.keys())
        if not day_keys:
            self.daily_chart.setTitle("Day-to-Day Detection by Class (no data)")
            return

        x_axis = QDateTimeAxis()
        x_axis.setFormat("MM-dd")
        x_axis.setTitleText("Date")
        x_axis.setTickCount(min(10, max(2, len(day_keys))))

        y_axis = QValueAxis()
        y_axis.setTitleText("Count")
        y_axis.setMin(0)

        max_y = 0
        for cls_name in CLASS_ORDER:
            series = QLineSeries()
            series.setName(cls_name)
            color = CLASS_COLORS.get(cls_name, Qt.black)
            pen = series.pen()
            pen.setColor(color)
            pen.setWidth(2)
            series.setPen(pen)

            for day_key in day_keys:
                dt = datetime.strptime(day_key, "%Y-%m-%d")
                x_ms = QDateTime(QDate(dt.year, dt.month, dt.day), QTime(0, 0)).toMSecsSinceEpoch()
                y_val = by_day[day_key].get(cls_name, 0)
                max_y = max(max_y, y_val)
                series.append(float(x_ms), float(y_val))
            self.daily_chart.addSeries(series)

        y_axis.setMax(max(3, max_y + 1))
        self.daily_chart.addAxis(x_axis, Qt.AlignBottom)
        self.daily_chart.addAxis(y_axis, Qt.AlignLeft)
        for s in self.daily_chart.series():
            s.attachAxis(x_axis)
            s.attachAxis(y_axis)
        self.daily_chart.legend().setVisible(True)
        self.daily_chart.setTitle("Day-to-Day Detection by Class")

    def _build_timeline_chart(self, events: List[EventItem]) -> None:
        self.timeline_chart.removeAllSeries()
        for ax in self.timeline_chart.axes():
            self.timeline_chart.removeAxis(ax)

        day = self.timeline_date_edit.date().toPython()
        day_events = [e for e in events if e.timestamp.date() == day]
        self.current_timeline_events = day_events
        if not day_events:
            self.timeline_chart.setTitle("Timeline Events (selected day) - no data")
            return

        day_start = datetime(day.year, day.month, day.day, 0, 0, 0)
        day_end = datetime(day.year, day.month, day.day, 23, 59, 59)
        x_axis = QDateTimeAxis()
        x_axis.setFormat("HH:mm")
        x_axis.setTitleText("Time")
        x_axis.setMin(QDateTime.fromMSecsSinceEpoch(int(day_start.timestamp() * 1000)))
        x_axis.setMax(QDateTime.fromMSecsSinceEpoch(int(day_end.timestamp() * 1000)))
        x_axis.setTickCount(9)

        y_axis = QCategoryAxis()
        y_axis.setRange(-0.5, float(len(CLASS_ORDER) - 0.5))
        for cls_name in CLASS_ORDER:
            y_axis.append(cls_name, float(self._timeline_class_to_index[cls_name]))
        y_axis.setLabelsPosition(QCategoryAxis.AxisLabelsPositionOnValue)
        y_axis.setTitleText("Class")

        for cls_name in CLASS_ORDER:
            series = QScatterSeries()
            series.setName(cls_name)
            series.setMarkerSize(10.0)
            color = CLASS_COLORS.get(cls_name, Qt.black)
            pen = series.pen()
            pen.setColor(color)
            series.setPen(pen)
            series.setColor(color)
            y_val = float(self._timeline_class_to_index[cls_name])
            for e in day_events:
                if e.event_type != cls_name:
                    continue
                x_ms = float(int(e.timestamp.timestamp() * 1000))
                series.append(x_ms, y_val)
            if series.count() > 0:
                series.clicked.connect(self._on_timeline_point_clicked)
                self.timeline_chart.addSeries(series)

        self.timeline_chart.addAxis(x_axis, Qt.AlignBottom)
        self.timeline_chart.addAxis(y_axis, Qt.AlignLeft)
        for s in self.timeline_chart.series():
            s.attachAxis(x_axis)
            s.attachAxis(y_axis)
        self.timeline_chart.legend().setVisible(True)
        self.timeline_chart.setTitle(f"Timeline Events ({day.strftime('%Y-%m-%d')})")

    def _fill_events_table(self, events: List[EventItem]) -> None:
        self.events_table.setRowCount(0)
        self.events_table.setSortingEnabled(False)
        for e in sorted(events, key=lambda x: x.timestamp, reverse=True):
            row = self.events_table.rowCount()
            self.events_table.insertRow(row)
            self.events_table.setItem(row, 0, QTableWidgetItem(e.timestamp.strftime("%Y-%m-%d %H:%M:%S")))
            self.events_table.setItem(row, 1, QTableWidgetItem(e.event_type))
            self.events_table.setItem(row, 2, QTableWidgetItem(e.camera))
            self.events_table.setItem(row, 3, QTableWidgetItem(e.zone_name))
            self.events_table.setItem(row, 4, QTableWidgetItem(e.person_id))
            self.events_table.setItem(row, 5, QTableWidgetItem(str(e.snapshot_path) if e.snapshot_path else "-"))
            self.events_table.setItem(row, 6, QTableWidgetItem(str(e.feedback_snapshot_path) if e.feedback_snapshot_path else "-"))
        self.events_table.setSortingEnabled(True)
        if self.events_table.rowCount() > 0:
            self.events_table.selectRow(0)
            self._on_event_row_selected(0, 0)
        else:
            self.snapshot_preview.setText("No snapshot selected")
            self.snapshot_preview.setPixmap(QPixmap())
            self.event_detail_text.setPlainText("")

    def _fill_summary_tables(self, events: List[EventItem]) -> None:
        daily = self._aggregate(events, key_fn=lambda e: e.date_key)
        monthly = self._aggregate(events, key_fn=lambda e: e.month_key)
        yearly = self._aggregate(events, key_fn=lambda e: e.year_key)
        self._apply_summary(self.daily_summary_table, sorted(daily.keys(), reverse=True), daily)
        self._apply_summary(self.monthly_summary_table, sorted(monthly.keys(), reverse=True), monthly)
        self._apply_summary(self.yearly_summary_table, sorted(yearly.keys(), reverse=True), yearly)

    def _aggregate(self, events: List[EventItem], key_fn) -> Dict[str, Dict[str, Any]]:
        out: Dict[str, Dict[str, Any]] = {}
        for e in events:
            key = key_fn(e)
            if key not in out:
                out[key] = {
                    "total": 0,
                    "haircut": 0,
                    "wash": 0,
                    "wait": 0,
                    "verified": 0,
                    "people": set(),
                }
            row = out[key]
            row["total"] += 1
            row[e.event_type] += 1
            if e.person_id:
                row["people"].add(e.person_id)
        return out

    def _apply_summary(self, table: QTableWidget, keys: List[str], agg: Dict[str, Dict[str, Any]]) -> None:
        table.setRowCount(0)
        for key in keys:
            item = agg[key]
            r = table.rowCount()
            table.insertRow(r)
            table.setItem(r, 0, QTableWidgetItem(key))
            table.setItem(r, 1, QTableWidgetItem(str(int(item["total"]))))
            table.setItem(r, 2, QTableWidgetItem(str(int(item["haircut"]))))
            table.setItem(r, 3, QTableWidgetItem(str(int(item["wash"]))))
            table.setItem(r, 4, QTableWidgetItem(str(int(item["wait"]))))
            table.setItem(r, 5, QTableWidgetItem(str(int(item["verified"]))))
            table.setItem(r, 6, QTableWidgetItem(str(len(item["people"]))))

    def _get_selected_event(self) -> Optional[EventItem]:
        row = self.events_table.currentRow()
        if row < 0:
            return None
        time_item = self.events_table.item(row, 0)
        event_type_item = self.events_table.item(row, 1)
        camera_item = self.events_table.item(row, 2)
        if not time_item or not event_type_item or not camera_item:
            return None
        ts_txt = time_item.text().strip()
        ev_type = event_type_item.text().strip().lower()
        camera = camera_item.text().strip()
        for e in self.events_filtered:
            if (
                e.timestamp.strftime("%Y-%m-%d %H:%M:%S") == ts_txt
                and e.event_type == ev_type
                and e.camera == camera
            ):
                return e
        return None

    def _on_event_row_selected(self, row: int, _col: int) -> None:
        if row < 0:
            return
        ev = self._get_selected_event()
        if ev is None:
            return
        self._show_event_detail(ev)

    def _show_event_detail(self, ev: EventItem) -> None:
        img_path = ev.snapshot_path if (ev.snapshot_path and ev.snapshot_path.exists()) else ev.feedback_snapshot_path
        if img_path and img_path.exists():
            pix = QPixmap(str(img_path))
            if not pix.isNull():
                self.snapshot_preview.setPixmap(
                    pix.scaled(self.snapshot_preview.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
                )
            else:
                self.snapshot_preview.setPixmap(QPixmap())
                self.snapshot_preview.setText("Failed to decode image")
        else:
            self.snapshot_preview.setPixmap(QPixmap())
            self.snapshot_preview.setText("Snapshot not found")

        detail = {
            "timestamp": ev.timestamp.isoformat(sep=" "),
            "event_type": ev.event_type,
            "camera": ev.camera,
            "zone": ev.zone_name,
            "person_id": ev.person_id,
            "snapshot_path": str(ev.snapshot_path) if ev.snapshot_path else "",
            "feedback_snapshot_path": str(ev.feedback_snapshot_path) if ev.feedback_snapshot_path else "",
            "metadata": ev.metadata,
        }
        self.event_detail_text.setPlainText(json.dumps(detail, ensure_ascii=False, indent=2))

    def resizeEvent(self, event):  # noqa: N802
        super().resizeEvent(event)
        ev = self._get_selected_event()
        if ev is not None:
            self._show_event_detail(ev)

    def open_selected_snapshot(self) -> None:
        ev = self._get_selected_event()
        if ev is None:
            QMessageBox.information(self, "Performance", "Please select an event first.")
            return
        img_path = ev.snapshot_path if (ev.snapshot_path and ev.snapshot_path.exists()) else ev.feedback_snapshot_path
        if img_path is None or not img_path.exists():
            QMessageBox.warning(self, "Performance", "Snapshot file not found for this event.")
            return
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(img_path.resolve())))

    def _on_timeline_point_clicked(self, point) -> None:
        if not self.current_timeline_events:
            return
        x = float(point.x())
        y = float(point.y())
        best = None
        best_score = None
        for e in self.current_timeline_events:
            cls_idx = float(self._timeline_class_to_index.get(e.event_type, -1))
            if abs(cls_idx - y) > 0.2:
                continue
            e_ms = float(int(e.timestamp.timestamp() * 1000))
            score = abs(e_ms - x)
            if best is None or score < best_score:
                best = e
                best_score = score
        if best is None:
            return
        target_ts = best.timestamp.strftime("%Y-%m-%d %H:%M:%S")
        for row in range(self.events_table.rowCount()):
            if (
                self.events_table.item(row, 0).text() == target_ts
                and self.events_table.item(row, 1).text().strip().lower() == best.event_type
                and self.events_table.item(row, 2).text().strip() == best.camera
            ):
                self.events_table.selectRow(row)
                self._show_event_detail(best)
                break
