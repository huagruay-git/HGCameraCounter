"""
Simple Zone Editor widget

Features implemented:
- Load snapshot from configured camera (via saved RTSP URL snapshot file or live capture)
- Draw polygon points with mouse clicks
- Move/delete points (basic)
- Name + type selection
- Save/Load zones_{camera}.json
- Basic validation (>=3 points, non-empty area, inside image bounds)
- Overlap detection (optional) - warns on overlap

This is a lightweight implementation intended to be integrated into the Controller UI.
"""

from pathlib import Path
import json
import math
import copy
from typing import List, Tuple, Dict, Optional

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QFileDialog,
    QComboBox, QLineEdit, QMessageBox, QListWidget, QListWidgetItem
)
from PySide6.QtGui import QPainter, QPen, QColor, QPixmap, QImage
from PySide6.QtCore import Qt, QPoint, QPointF, QRect, QThread, Signal
import cv2


ZoneTypeChoices = ["CHAIR", "WAIT", "WASH", "STAFF_AREA", "OTHER"]


class _SnapshotLoader(QThread):
    """Open an RTSP URL and grab ONE frame OFF the UI thread (with a connect timeout),
    so the window never freezes ('Not Responding') while a camera connects."""
    done = Signal(bool, object, int, int, str)  # ok, frame(ndarray|None), w, h, error

    def __init__(self, url: str):
        super().__init__()
        self.url = url

    def run(self):
        cap = None
        try:
            cap = cv2.VideoCapture(self.url, cv2.CAP_FFMPEG)
            for prop in ("CAP_PROP_OPEN_TIMEOUT_MSEC", "CAP_PROP_READ_TIMEOUT_MSEC"):
                if hasattr(cv2, prop):
                    try:
                        cap.set(getattr(cv2, prop), 8000)
                    except Exception:
                        pass
            ok, frame = cap.read()
            if not ok or frame is None:
                self.done.emit(False, None, 0, 0, "no frame (connect failed / timeout)")
                return
            h, w = frame.shape[:2]
            self.done.emit(True, frame, w, h, "")
        except Exception as e:
            self.done.emit(False, None, 0, 0, str(e))
        finally:
            try:
                if cap is not None:
                    cap.release()
            except Exception:
                pass


class PolygonCanvas(QWidget):
    """Canvas to draw image and polygons"""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.image = None  # QImage
        self._image_dims = (1,1)  # (width, height)
        self._draw_rect = QRect(0, 0, 1, 1)
        self.polygons: List[List[QPointF]] = []
        self.current: List[QPointF] = []
        # Called after the user drags a polygon point, so the editor can write the new
        # positions back into its zone store (otherwise reshape edits are lost on save).
        self.on_polygons_edited = None
        self.selected_idx = None
        self._dragging = False
        self._drag_poly: Optional[int] = None
        self._drag_point: Optional[int] = None
        self.undo_stack: List[Dict] = []
        self.last_finished: List[QPointF] = []
        # Allow a smaller editor window on compact displays
        self.setMinimumSize(320, 180)

    def load_image(self, qimage: QImage):
        self.image = qimage
        self._image_dims = (qimage.width(), qimage.height())
        self.update()

    def _update_draw_rect(self):
        """Calculate actual image draw rect (with KeepAspectRatio letterbox offsets)."""
        if not self.image:
            self._draw_rect = self.rect()
            return
        iw, ih = self.image.width(), self.image.height()
        if iw <= 0 or ih <= 0 or self.width() <= 0 or self.height() <= 0:
            self._draw_rect = self.rect()
            return
        scale = min(self.width() / float(iw), self.height() / float(ih))
        draw_w = max(1, int(iw * scale))
        draw_h = max(1, int(ih * scale))
        x = (self.width() - draw_w) // 2
        y = (self.height() - draw_h) // 2
        self._draw_rect = QRect(x, y, draw_w, draw_h)

    def _to_pixel_coords(self, norm_x: float, norm_y: float) -> QPoint:
        self._update_draw_rect()
        x = self._draw_rect.x() + int(norm_x * self._draw_rect.width())
        y = self._draw_rect.y() + int(norm_y * self._draw_rect.height())
        return QPoint(x, y)
    
    def _to_norm_coords(self, pixel_x: int, pixel_y: int) -> QPointF:
        self._update_draw_rect()
        if self._draw_rect.width() <= 0 or self._draw_rect.height() <= 0:
            return QPointF(0.0, 0.0)
        nx = (pixel_x - self._draw_rect.x()) / float(self._draw_rect.width())
        ny = (pixel_y - self._draw_rect.y()) / float(self._draw_rect.height())
        nx = max(0.0, min(1.0, nx))
        ny = max(0.0, min(1.0, ny))
        return QPointF(nx, ny)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor(30, 30, 30))
        if self.image:
            self._update_draw_rect()
            pix = QPixmap.fromImage(self.image).scaled(
                self._draw_rect.size(),
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation
            )
            painter.drawPixmap(self._draw_rect.topLeft(), pix)
        # draw existing polygons
        pen = QPen(QColor(0, 255, 0), 2)
        painter.setPen(pen)
        for poly in self.polygons:
            if len(poly) >= 2:
                for i in range(len(poly)):
                    p1_pixel  = self._to_pixel_coords(poly[i].x(), poly[i].y())
                    p2_pixel = self._to_pixel_coords(poly[(i + 1) % len(poly)].x(), poly[(i + 1) % len(poly)].y())
                    painter.drawLine(p1_pixel, p2_pixel)
            for p in poly:
                p_pixel = self._to_pixel_coords(p.x(), p.y())
                painter.drawEllipse(p_pixel, 4, 4)
        # draw current
        pen = QPen(QColor(255, 165, 0), 2)
        painter.setPen(pen)
        for i in range(len(self.current)):
            p1_pixel = self._to_pixel_coords(self.current[i].x(), self.current[i].y())
            if i + 1 < len(self.current):
                p2_pixel = self._to_pixel_coords(self.current[i + 1].x(), self.current[i + 1].y())
                painter.drawLine(p1_pixel, p2_pixel)
            painter.drawEllipse(p1_pixel, 4, 4)


    def mousePressEvent(self, event):
        pos_pixel = event.position().toPoint()
        self._update_draw_rect()
        if not self._draw_rect.contains(pos_pixel):
            return
        pos_norm = self._to_norm_coords(pos_pixel.x(), pos_pixel.y())
        if event.button() == Qt.LeftButton:
            # hit-test existing points for dragging
            found = False
            for pi, poly in enumerate(self.polygons):
                for pj, p_norm in enumerate(poly):
                    p_pixel = self._to_pixel_coords(p_norm.x(), p_norm.y())
                    if (p_pixel - pos_pixel).manhattanLength() <= 8:
                        self._dragging = True
                        self._drag_poly = pi
                        self._drag_point = pj
                        found = True
                        break
                if found:
                    break
            if not found:
                # add point to current polygon
                self.current.append(pos_norm)
            self.update()
        elif event.button() == Qt.RightButton:
            # finish polygon (keep copy so editor can create zone from it)
            if len(self.current) >= 3:
                self.last_finished = self.current.copy()
                self.polygons.append(self.current.copy())
                # save undo snapshot
                try:
                    self.push_undo()
                except Exception:
                    pass
                self.current = []
                self.update()

    def mouseMoveEvent(self, event):
        if self._dragging and self._drag_poly is not None and self._drag_point is not None:
            pos_pixel = event.position().toPoint()
            self._update_draw_rect()
            if not self._draw_rect.contains(pos_pixel):
                return
            pos_norm = self._to_norm_coords(pos_pixel.x(), pos_pixel.y())
            try:
                self.polygons[self._drag_poly][self._drag_point] = pos_norm
                self.update()
            except Exception:
                pass

    def mouseReleaseEvent(self, event):
        if self._dragging:
            # push undo snapshot
            try:
                self.push_undo()
            finally:
                self._dragging = False
                self._drag_poly = None
                self._drag_point = None
            cb = getattr(self, "on_polygons_edited", None)
            if callable(cb):
                cb()

    def mouseDoubleClickEvent(self, event):
        # double-click near a point to delete it
        pos_pixel = event.position().toPoint()
        self._update_draw_rect()
        if not self._draw_rect.contains(pos_pixel):
            return
        pos_norm = self._to_norm_coords(pos_pixel.x(), pos_pixel.y())
        for pi, poly in enumerate(self.polygons):
            for pj, p_norm in enumerate(poly):
                p_pixel = self._to_pixel_coords(p_norm.x(), p_norm.y())
                if (p_pixel - pos_pixel).manhattanLength() <= 8:
                    # remove point
                    del self.polygons[pi][pj]
                    # if polygon becomes too small, remove polygon
                    if len(self.polygons[pi]) < 3:
                        del self.polygons[pi]
                    self.push_undo()
                    self.update()
                    return

    def push_undo(self):
        # store deep copy of polygons and current
        self.undo_stack.append({
            "polygons": copy.deepcopy(self.polygons),
            "current": copy.deepcopy(self.current),
            "last_finished": copy.deepcopy(self.last_finished),
        })

    def undo(self):
        if not self.undo_stack:
            return
        snap = self.undo_stack.pop()
        self.polygons = snap.get("polygons", [])
        self.current = snap.get("current", [])
        self.last_finished = snap.get("last_finished", [])
        self.update()

    def clear_current(self):
        """Clear current in-progress polygon."""
        if self.current:
            self.push_undo()
        self.current = []
        self.last_finished = []
        self.update()

    def clear_all(self):
        """Clear all polygons and current drawing."""
        if self.polygons or self.current:
            self.push_undo()
        self.polygons = []
        self.current = []
        self.last_finished = []
        self.selected_idx = None
        self.update()


class ZoneEditorWidget(QWidget):
    def __init__(self, main_controller=None):
        super().__init__()
        self.controller = main_controller
        self.config = main_controller.config if main_controller else {}
        self.canvas = PolygonCanvas(self)
        self.zones: Dict[str, List[Dict]] = {}  # camera_name -> list[zones]
        self.current_camera = None
        self._canvas_zone_index: List[int] = []
        # Live-sync dragged points back into self.zones the moment a drag finishes.
        self.canvas.on_polygons_edited = self._sync_canvas_edits_to_zones

        self.init_ui()

    def _normalized_points(self, raw_points: List, image_size: Optional[Tuple[int, int]] = None) -> List[List[float]]:
        """Convert points from mixed formats (normalized/pixel) into normalized [x,y]."""
        if not raw_points:
            return []

        pts: List[List[float]] = []
        for p in raw_points:
            if isinstance(p, dict):
                x = float(p.get("x", 0.0))
                y = float(p.get("y", 0.0))
            else:
                x = float(p[0])
                y = float(p[1])
            pts.append([x, y])

        # Already normalized.
        if all(0.0 <= x <= 1.0 and 0.0 <= y <= 1.0 for x, y in pts):
            return pts

        # Pixel -> normalized (prefer actual image size if available).
        w = h = None
        if image_size:
            w, h = image_size
        elif self.canvas.image:
            w, h = self.canvas.image.width(), self.canvas.image.height()

        if not w or not h:
            max_x = max(x for x, _ in pts) if pts else 1.0
            max_y = max(y for _, y in pts) if pts else 1.0
            w = max(max_x, 1.0)
            h = max(max_y, 1.0)

        return [[x / float(w), y / float(h)] for x, y in pts]

    def _refresh_canvas_polygons(self):
        """Render all zones for selected camera on the canvas."""
        cam = self.current_camera or self.camera_selector.currentText()
        if not cam:
            self.canvas.polygons = []
            self.canvas.update()
            return
        polys, idx_map = [], []
        for i, zone in enumerate(self.zones.get(cam, [])):
            pts = zone.get("points", [])
            if len(pts) >= 3:
                polys.append([QPointF(p[0], p[1]) for p in pts])
                idx_map.append(i)
        self.canvas.polygons = polys
        # Remember which self.zones entry each canvas polygon came from, so dragged
        # points can be written back to the right zone on save.
        self._canvas_zone_index = idx_map
        self.canvas.update()

    def _sync_canvas_edits_to_zones(self):
        """Write point positions edited on the canvas (dragging) back into self.zones,
        so 'Save Zones' persists reshape edits. Maps each canvas polygon to its zone via
        the index recorded in _refresh_canvas_polygons. Skips safely if the lists are out
        of sync (e.g. a whole polygon was just deleted on the canvas)."""
        cam = self.current_camera
        if not cam or cam not in self.zones:
            return
        polys = getattr(self.canvas, "polygons", [])
        idx_map = getattr(self, "_canvas_zone_index", [])
        if len(idx_map) != len(polys):
            return
        zones = self.zones[cam]
        for poly, zi in zip(polys, idx_map):
            if 0 <= zi < len(zones) and len(poly) >= 3:
                zones[zi]["points"] = [[p.x(), p.y()] for p in poly]

    def init_ui(self):
        layout = QVBoxLayout()
        self.help_label = QLabel(
            "Workflow: Select camera -> Load Snapshot -> Left click to add points -> "
            "Create/Update Zone (drag point to move, double-click point to delete)"
        )
        self.help_label.setWordWrap(True)
        self.help_label.setStyleSheet("color: #666;")
        layout.addWidget(self.help_label)
        hl = QHBoxLayout()
        # Camera selector
        hl.addWidget(QLabel("Camera:"))
        self.camera_selector = QComboBox()
        self.reload_cameras()
        hl.addWidget(self.camera_selector, 1)
        self.camera_selector.currentTextChanged.connect(self.on_camera_selected)

        reload_cam_btn = QPushButton("Reload Cameras")
        reload_cam_btn.clicked.connect(self.reload_cameras)
        hl.addWidget(reload_cam_btn)
        
        self.load_btn = QPushButton("Load Snapshot")
        self.load_btn.clicked.connect(self.load_snapshot_from_selected)
        hl.addWidget(self.load_btn)

        undo_btn = QPushButton("Undo")
        undo_btn.clicked.connect(self.canvas.undo)
        hl.addWidget(undo_btn)

        save_btn = QPushButton("Save Zones")
        save_btn.clicked.connect(self.save_zones)
        hl.addWidget(save_btn)

        loadzones_btn = QPushButton("Load Zones")
        loadzones_btn.clicked.connect(self.load_zones)
        hl.addWidget(loadzones_btn)

        layout.addLayout(hl)
        layout.addWidget(self.canvas, 1)

        # Zone properties
        prop_h = QHBoxLayout()
        prop_h.addWidget(QLabel("Zone Name:"))
        self.zone_name = QLineEdit()
        self.zone_name.setPlaceholderText("e.g. CHAIR_01")
        self.zone_name.returnPressed.connect(self.create_zone_from_current)
        prop_h.addWidget(self.zone_name)
        prop_h.addWidget(QLabel("Type:"))
        self.zone_type = QComboBox()
        self.zone_type.addItems(ZoneTypeChoices)
        prop_h.addWidget(self.zone_type)

        add_zone_btn = QPushButton("Create / Update Zone")
        add_zone_btn.clicked.connect(self.create_zone_from_current)
        prop_h.addWidget(add_zone_btn)

        new_zone_btn = QPushButton("New Zone")
        new_zone_btn.clicked.connect(self.start_new_zone)
        prop_h.addWidget(new_zone_btn)

        clear_current_btn = QPushButton("Clear Current")
        clear_current_btn.clicked.connect(self.clear_current_points)
        prop_h.addWidget(clear_current_btn)

        clear_all_btn = QPushButton("Clear All")
        clear_all_btn.clicked.connect(self.clear_all_points)
        prop_h.addWidget(clear_all_btn)

        layout.addLayout(prop_h)

        info_row = QHBoxLayout()
        self.zone_count_label = QLabel("Zones: 0")
        info_row.addWidget(self.zone_count_label)
        info_row.addStretch()
        layout.addLayout(info_row)

        # Zone list
        self.zone_list = QListWidget()
        self.zone_list.setAlternatingRowColors(True)
        self.zone_list.currentRowChanged.connect(self.on_zone_selected)
        layout.addWidget(self.zone_list)

        # Zone actions
        zone_actions = QHBoxLayout()
        edit_zone_btn = QPushButton("Edit Selected Zone")
        edit_zone_btn.clicked.connect(self.edit_selected_zone)
        zone_actions.addWidget(edit_zone_btn)
        del_zone_btn = QPushButton("Delete Selected Zone")
        del_zone_btn.clicked.connect(self.delete_selected_zone)
        zone_actions.addWidget(del_zone_btn)
        zone_actions.addStretch()
        layout.addLayout(zone_actions)

        self.status_label = QLabel("Ready")
        self.status_label.setWordWrap(True)
        self.status_label.setStyleSheet("color: #2e7d32;")
        layout.addWidget(self.status_label)

        self.setLayout(layout)
        self.start_new_zone()

    def _set_status(self, message: str, is_error: bool = False):
        color = "#c62828" if is_error else "#2e7d32"
        if hasattr(self, "status_label"):
            self.status_label.setStyleSheet(f"color: {color};")
            self.status_label.setText(message)

    def _refresh_zone_list_ui(self, cam: Optional[str] = None):
        target_cam = cam or self.current_camera or self.camera_selector.currentText()
        self.zone_list.clear()
        if not target_cam:
            self.zone_count_label.setText("Zones: 0")
            return
        cam_zones = self.zones.get(target_cam, [])
        for i, z in enumerate(cam_zones, start=1):
            name = z.get("name", "")
            ztype = z.get("type", "")
            self.zone_list.addItem(QListWidgetItem(f"{i}. {name} ({ztype})"))
        self.zone_count_label.setText(f"Zones: {len(cam_zones)}")

    def _suggest_zone_name(self) -> str:
        cam = self.current_camera or self.camera_selector.currentText()
        zone_type = self.zone_type.currentText() if hasattr(self, "zone_type") else "ZONE"
        prefix = zone_type or "ZONE"
        existing = self.zones.get(cam, []) if cam else []
        return f"{prefix}_{len(existing) + 1:02d}"

    def _active_points_for_zone(self) -> List[QPointF]:
        if len(self.canvas.current) >= 3:
            return list(self.canvas.current)
        if len(self.canvas.last_finished) >= 3:
            return list(self.canvas.last_finished)
        return []

    def start_new_zone(self):
        self.zone_list.blockSignals(True)
        self.zone_list.clearSelection()
        self.zone_list.setCurrentRow(-1)
        self.zone_list.blockSignals(False)
        self.canvas.clear_current()
        self.zone_name.setText(self._suggest_zone_name())
        self._set_status("Ready to draw a new zone")

    def clear_current_points(self):
        self.canvas.clear_current()
        self._set_status("Cleared current drawing")

    def clear_all_points(self):
        self.canvas.clear_all()
        self._set_status("Cleared all polygons on canvas")

    def reload_cameras(self):
        prev = self.camera_selector.currentText() if hasattr(self, "camera_selector") else ""
        cameras = self.config.get("cameras", {}) or {}
        self.camera_selector.blockSignals(True)
        self.camera_selector.clear()
        for name in cameras.keys():
            self.camera_selector.addItem(name)
        if prev and prev in cameras:
            self.camera_selector.setCurrentText(prev)
        elif self.current_camera and self.current_camera in cameras:
            self.camera_selector.setCurrentText(self.current_camera)
        self.camera_selector.blockSignals(False)
        if cameras:
            self._set_status(f"Cameras loaded: {len(cameras)}")
        else:
            self._set_status("No cameras configured", is_error=True)

    def on_camera_selected(self, index):
        cam = self.camera_selector.currentText()
        if cam:
            self.current_camera = cam
            self.canvas.clear_all()  # clear previous camera drawing
            self.load_zones(show_notice=False)  # auto load zones for selected camera
            self.start_new_zone()
            self._set_status(f"Selected camera: {cam}")

    def load_snapshot_from_selected(self):
        cam = self.camera_selector.currentText()
        if not cam:
            QMessageBox.warning(self, "Error", "No camera selected")
            self._set_status("No camera selected", is_error=True)
            return
        self.current_camera = cam
        cam_conf = self.config.get("cameras", {}).get(cam, {})
        url = cam_conf.get("rtsp_url")
        if not url:
            QMessageBox.warning(self, "Error", "RTSP URL not configured for this camera")
            self._set_status(f"RTSP URL missing for {cam}", is_error=True)
            return
        # Grab the frame OFF the UI thread so the window stays responsive.
        if getattr(self, "_snap_loader", None) is not None and self._snap_loader.isRunning():
            self._set_status("Still loading the previous snapshot...")
            return
        if hasattr(self, "load_btn"):
            self.load_btn.setEnabled(False)
        self._set_status(f"Loading snapshot from {cam}...")
        self._snap_loader = _SnapshotLoader(url)
        self._snap_loader.done.connect(self._on_snapshot_loaded)
        self._snap_loader.start()

    def _on_snapshot_loaded(self, ok: bool, frame, w: int, h: int, error: str):
        if hasattr(self, "load_btn"):
            self.load_btn.setEnabled(True)
        if not ok or frame is None:
            QMessageBox.critical(self, "Error", f"Cannot capture frame: {error}")
            self._set_status(f"Snapshot failed: {error}", is_error=True)
            return
        try:
            ch = frame.shape[2] if frame.ndim == 3 else 1
            # .copy() so the QImage owns its pixels (the ndarray is freed after this slot).
            qimg = QImage(frame.data, w, h, ch * w, QImage.Format_BGR888).copy()
            self.canvas.load_image(qimg)
            cam_zones = self.zones.get(self.current_camera, [])
            if cam_zones:
                for z in cam_zones:
                    z["points"] = self._normalized_points(z.get("points", []), image_size=(w, h))
                self._refresh_canvas_polygons()
            self._set_status(f"Snapshot loaded for {self.current_camera} ({w}x{h})")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Snapshot render failed: {e}")
            self._set_status(f"Snapshot render failed: {e}", is_error=True)

    def create_zone_from_current(self):
        name = self.zone_name.text().strip()
        if not name:
            QMessageBox.warning(self, "Error", "Zone name required")
            self._set_status("Zone name required", is_error=True)
            return
        pts_norm = self._active_points_for_zone()
        if len(pts_norm) < 3:
            QMessageBox.warning(self, "Error", "Polygon must have at least 3 points")
            self._set_status("Polygon must have at least 3 points", is_error=True)
            return

        for p in pts_norm:
            if not (0.0 <= p.x() <= 1.0 and 0.0 <= p.y() <= 1.0):
                QMessageBox.warning(self, "Error", "Polygon points must be inside image bounds")
                self._set_status("Polygon points must be inside image bounds", is_error=True)
                return

        z = {
            "name": name,
            "type": self.zone_type.currentText(),
            "points": [[p.x(), p.y()] for p in pts_norm]
        }
        if self.current_camera is None:
            QMessageBox.warning(self, "Error", "No camera loaded")
            self._set_status("No camera loaded", is_error=True)
            return
        
        # Check if editing an existing zone or creating a new one
        selected_row = self.zone_list.currentRow()
        cam_zones = self.zones.setdefault(self.current_camera, [])
        can_edit = 0 <= selected_row < len(cam_zones)

        if can_edit:
            cam_zones[selected_row] = z
            status_message = f"Updated zone '{name}'"
            target_row = selected_row
        else:
            cam_zones.append(z)
            status_message = f"Created zone '{name}'"
            target_row = len(cam_zones) - 1

        self._refresh_zone_list_ui(self.current_camera)
        self.zone_list.setCurrentRow(target_row)
        self._refresh_canvas_polygons()
        self.canvas.clear_current()
        self.canvas.last_finished = []
        self.canvas.update()
        self._set_status(status_message)

        # optional overlap check
        if self._check_overlap(self.current_camera, z):
            QMessageBox.warning(self, "Warning", "Zone overlaps existing zone")
            self._set_status("Warning: Zone overlaps existing zone", is_error=True)

    def save_zones(self):
        if not self.current_camera:
            QMessageBox.warning(self, "Error", "No camera selected/loaded")
            self._set_status("No camera selected", is_error=True)
            return
        # Pull any on-canvas point drags into self.zones before writing the file.
        self._sync_canvas_edits_to_zones()
        zones = self.zones.get(self.current_camera, [])
        if not zones:
            self._set_status("No zones to save", is_error=True)
            return
        zones_to_save = []
        for z in zones:
            pts = self._normalized_points(z.get("points", []))
            zones_to_save.append({
                "name": z.get("name", ""),
                "type": z.get("type", "OTHER"),
                "points": pts,
                "polygon_json": [{"x": p[0], "y": p[1]} for p in pts],
            })
        zones_dir = Path(self.config.get("paths", {}).get("zones", "data/zones"))
        zones_dir.mkdir(parents=True, exist_ok=True)
        file_path = zones_dir / f"zones_{self.current_camera}.json"
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(zones_to_save, f, indent=2, ensure_ascii=False)
            self._set_status(f"Saved {len(zones_to_save)} zone(s) to {file_path}")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to save zones: {e}")
            self._set_status(f"Failed to save zones: {e}", is_error=True)

    def load_zones(self, show_notice: bool = True):
        cam = self.camera_selector.currentText()
        if not cam:
            QMessageBox.warning(self, "Error", "No camera selected")
            self._set_status("No camera selected", is_error=True)
            return
        self.current_camera = cam
        zones_dir = Path(self.config.get("paths", {}).get("zones", "data/zones"))
        file_path = zones_dir / f"zones_{cam}.json"
        if not file_path.exists():
            self.zones[cam] = []
            self._refresh_zone_list_ui(cam)
            self._refresh_canvas_polygons()
            if show_notice:
                self._set_status(f"No zones file for {cam}")
            return
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                loaded = json.load(f)
            normalized_loaded = []
            for z in loaded:
                pts = z.get("points")
                if not pts and "polygon_json" in z:
                    pts = z.get("polygon_json", [])
                norm_pts = self._normalized_points(pts or [])
                zone_type = z.get("type", "OTHER")
                zone_name = z.get("name", "")
                if zone_type == "OTHER":
                    up = zone_name.upper()
                    if "CHAIR" in up:
                        zone_type = "CHAIR"
                    elif "WAIT" in up:
                        zone_type = "WAIT"
                    elif "WASH" in up:
                        zone_type = "WASH"
                    elif "STAFF" in up:
                        zone_type = "STAFF_AREA"

                normalized_loaded.append({
                    "name": zone_name,
                    "type": zone_type,
                    "points": norm_pts,
                })

            self.zones[cam] = normalized_loaded
            self._refresh_zone_list_ui(cam)
            self._refresh_canvas_polygons()
            if show_notice:
                self._set_status(f"Loaded {len(loaded)} zone(s) for {cam}")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load zones: {e}")
            self._set_status(f"Failed to load zones: {e}", is_error=True)

    def delete_selected_zone(self):
        cam = self.current_camera or self.camera_selector.currentText()
        idx = self.zone_list.currentRow()
        if idx < 0:
            self._set_status("No zone selected", is_error=True)
            return
        if not cam or cam not in self.zones:
            QMessageBox.warning(self, "Error", "No zones loaded for this camera")
            self._set_status("No zones loaded for this camera", is_error=True)
            return
        try:
            zone_name = self.zones[cam][idx].get("name", f"#{idx+1}")
            del self.zones[cam][idx]
            self._refresh_zone_list_ui(cam)
            self._refresh_canvas_polygons()
            self.start_new_zone()
            self._set_status(f"Deleted zone '{zone_name}'")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to delete zone: {e}")
            self._set_status(f"Failed to delete zone: {e}", is_error=True)

    def on_zone_selected(self, row: int):
        """Load selected zone into form and canvas preview."""
        cam = self.current_camera or self.camera_selector.currentText()
        if row < 0 or not cam or cam not in self.zones:
            return
        if row >= len(self.zones[cam]):
            return

        zone = self.zones[cam][row]
        self.zone_name.setText(zone.get("name", ""))
        zone_type = zone.get("type", "OTHER")
        idx = self.zone_type.findText(zone_type)
        if idx >= 0:
            self.zone_type.setCurrentIndex(idx)

        self.canvas.current = [QPointF(p[0], p[1]) for p in zone.get("points", [])]
        self.canvas.last_finished = list(self.canvas.current)
        self._refresh_canvas_polygons()
        self.canvas.update()
        self._set_status(f"Editing zone '{zone.get('name', '')}'")

    def edit_selected_zone(self):
        """Prepare selected zone for editing by loading its points."""
        row = self.zone_list.currentRow()
        if row < 0:
            self._set_status("No zone selected", is_error=True)
            return
        self.on_zone_selected(row)
        self._set_status("Zone loaded. Move points and click Create / Update Zone")

    def _check_overlap(self, camera: str, new_zone: Dict) -> bool:
        """Check polygon overlap using segment intersection and point-in-polygon tests."""
        def seg_intersect(a1, a2, b1, b2):
            # Check if segments a1-a2 and b1-b2 intersect (using pixel coordinates)
            def orient(p, q, r):
                return (q[0]-p[0])*(r[1]-p[1]) - (q[1]-p[1])*(r[0]-p[0])
            
            def on_segment(p, q, r):
                return (min(p[0], r[0]) <= q[0] <= max(p[0], r[0]) and
                        min(p[1], r[1]) <= q[1] <= max(p[1], r[1]))

            o1 = orient(a1, a2, b1)
            o2 = orient(a1, a2, b2)
            o3 = orient(b1, b2, a1)
            o4 = orient(b1, b2, a2)

            if o1 == 0 and on_segment(a1, b1, a2):
                return True
            if o2 == 0 and on_segment(a1, b2, a2):
                return True
            if o3 == 0 and on_segment(b1, a1, b2):
                return True
            if o4 == 0 and on_segment(b1, a2, b2):
                return True

            return (o1 * o2 < 0) and (o3 * o4 < 0)
        
        def point_in_poly(pt: QPointF, poly: List[QPointF]) -> bool:
            # ray casting algorithm
            x, y = pt.x(), pt.y()
            inside = False
            n = len(poly)
            for i in range(n):
                p1 = poly[i]
                p2 = poly[(i+1) % n]
                if ((p1.y() > y) != (p2.y() > y)) and (x < (p2.x()-p1.x()) * (y - p1.y()) / (p2.y() - p1.y() + 1e-12) + p1.x()):
                    inside = not inside
            return inside

        new_pts_norm = [QPointF(p[0], p[1]) for p in new_zone['points']]

        for z in self.zones.get(camera, []):
            # Skip checking overlap with itself if editing
            selected_row = self.zone_list.currentRow()
            if selected_row >= 0 and self.zones[camera][selected_row] is z:
                continue

            # Convert stored normalized points to pixel for comparison
            pts_norm = [QPointF(p[0], p[1]) for p in z['points']]
            # segment intersection check
            for i in range(len(new_pts_norm)):
                a1 = new_pts_norm[i]
                a2 = new_pts_norm[(i+1) % len(new_pts_norm)]
                for j in range(len(pts_norm)):
                    b1 = pts_norm[j]
                    b2 = pts_norm[(j+1) % len(pts_norm)]
                    if seg_intersect(a1, a2, b1, b2):
                        return True
            # vertex inside other polygon
            if point_in_poly(new_pts_norm[0], pts_norm) or point_in_poly(pts_norm[0], new_pts_norm):
                return True
        return False

    def validate_zones(self, camera: str) -> Tuple[bool, str]:
        zones = self.zones.get(camera, [])
        if not zones:
            return False, "No zones"
        for z in zones:
            if len(z.get('points', [])) < 3:
                return False, f"Zone {z.get('name','?')} has fewer than 3 points"
        return True, "OK"


