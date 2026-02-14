from pathlib import Path
import json
import os
from typing import Dict, List

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QLineEdit, QFileDialog, QTextEdit
)
from PySide6.QtCore import Qt
import cv2
import numpy as np


class StaffBuilderWidget(QWidget):
    """Simple Staff DB builder UI

    - Select `staff_gallery` folder
    - Scan subfolders (one folder per staff)
    - Check image count, size, blur (Laplacian var)
    - Build `staff_db.json` with list of staff and image paths
    """

    def __init__(self, parent=None, config=None):
        super().__init__(parent)
        self.config = config or {}
        self.gallery_path = ''
        self.report = []
        self.staff_db: Dict[str, List[str]] = {}
        self.init_ui()

    def _cfg_get(self, key: str, default=None):
        """Read config values from dict-like or Config object."""
        if isinstance(self.config, dict):
            return self.config.get(key, default)
        if hasattr(self.config, "get"):
            try:
                return self.config.get(key, default)
            except Exception:
                return default
        return default

    def _default_gallery_path(self) -> str:
        """Pick configured gallery path; fallback to legacy folder if needed."""
        cfg_paths = self._cfg_get("paths", {}) or {}
        preferred = Path(cfg_paths.get("staff_gallery", "data/staff_gallery"))
        legacy = Path("staff_gallery")

        if preferred.exists() and any(preferred.iterdir()):
            return str(preferred)
        if legacy.exists() and any(legacy.iterdir()):
            return str(legacy)
        return str(preferred)

    def _staff_db_output_path(self) -> Path:
        cfg_paths = self._cfg_get("paths", {}) or {}
        return Path(cfg_paths.get("staff_db", "data/staff_gallery/staff_db.json"))

    def init_ui(self):
        layout = QVBoxLayout()

        hl = QHBoxLayout()
        hl.addWidget(QLabel("staff_gallery folder:"))
        self.path_edit = QLineEdit()
        self.path_edit.setText(self._default_gallery_path())
        hl.addWidget(self.path_edit)
        browse = QPushButton("Browse")
        browse.clicked.connect(self.browse_folder)
        hl.addWidget(browse)
        layout.addLayout(hl)

        actions = QHBoxLayout()
        scan_btn = QPushButton("Scan")
        scan_btn.clicked.connect(self.scan_gallery)
        actions.addWidget(scan_btn)

        build_btn = QPushButton("Build staff_db.json")
        build_btn.clicked.connect(self.build_staff_db)
        actions.addWidget(build_btn)

        layout.addLayout(actions)

        self.output = QTextEdit()
        self.output.setReadOnly(True)
        layout.addWidget(self.output)

        self.setLayout(layout)

    def browse_folder(self):
        dlg = QFileDialog(self)
        dlg.setFileMode(QFileDialog.Directory)
        if dlg.exec():
            folders = dlg.selectedFiles()
            if folders:
                self.path_edit.setText(folders[0])

    def scan_gallery(self):
        path = self.path_edit.text().strip()
        if not path:
            self.output.setPlainText("Please select staff_gallery folder")
            return
        p = Path(path)
        if not p.exists() or not p.is_dir():
            self.output.setPlainText("Invalid folder")
            return

        self.staff_db = {}
        report_lines = []
        total_images = 0

        for person_dir in sorted(p.iterdir()):
            if not person_dir.is_dir():
                continue
            name = person_dir.name
            imgs = []
            person_report = []
            for img_file in sorted(person_dir.iterdir()):
                if img_file.suffix.lower() not in ('.jpg', '.jpeg', '.png', '.bmp'):
                    continue
                total_images += 1
                size_ok = img_file.stat().st_size > 1024  # >1KB
                blur_ok = True
                try:
                    img = cv2.imdecode(np.fromfile(str(img_file), dtype='uint8'), cv2.IMREAD_GRAYSCALE)
                    if img is None:
                        blur_ok = False
                    else:
                        var = cv2.Laplacian(img, cv2.CV_64F).var()
                        blur_ok = var > 100.0
                except Exception:
                    blur_ok = False

                reason = []
                if not size_ok:
                    reason.append('small file')
                if not blur_ok:
                    reason.append('blurry or unreadable')
                if reason:
                    person_report.append(f"{img_file.name}: FAIL ({', '.join(reason)})")
                else:
                    person_report.append(f"{img_file.name}: OK")
                    imgs.append(str(img_file))

            self.staff_db[name] = imgs
            report_lines.append(f"{name}: {len(imgs)} valid images")
            report_lines.extend(person_report)

        report_lines.insert(0, f"Total persons: {len(self.staff_db)}, Total images scanned: {total_images}")
        self.output.setPlainText('\n'.join(report_lines))

    def build_staff_db(self):
        gallery = self.path_edit.text().strip() or self._default_gallery_path()
        gpath = Path(gallery)
        if not gpath.exists() or not gpath.is_dir():
            self.output.setPlainText(f"Invalid staff gallery path: {gallery}")
            return

        out_file = self._staff_db_output_path()
        out_file.parent.mkdir(parents=True, exist_ok=True)

        self.output.append(f"Building staff DB from: {gpath}")
        self.output.append(f"Output: {out_file}")
        try:
            from runtime.build_staff_db import build_staff_db as build_staff_db_embeddings
            report = build_staff_db_embeddings(
                staff_gallery_dir=str(gpath),
                output_path=str(out_file),
                save_crops=True,
            )
            self.output.append(
                f"Done. staff={report.total_staff}, images={report.total_images}, "
                f"success={report.success_count}, failed={len(report.failed_images)}"
            )
            if report.failed_images:
                self.output.append("Some images failed; check logs/build_staff_db.log")
        except Exception as e:
            self.output.append(f"Build failed: {e}")
