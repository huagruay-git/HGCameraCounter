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
from PySide6.QtCore import Qt, QPoint, QPointF, QRect
import cv2


ZoneTypeChoices = ["CHAIR", "WAIT", "WASH", "STAFF_AREA", "OTHER"]


class PolygonCanvas(QWidget):
    """Canvas to draw image and polygons"""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.image = None  # QImage
        self._image_dims = (1,1)  # (width, height)
        self._draw_rect = QRect(0, 0, 1, 1)
        self.polygons: List[List[QPointF]] = []
        self.current: List[QPointF] = []
        self.selected_idx = None
        self._dragging = False
        self._drag_poly: Optional[int] = None
        self._drag_point: Optional[int] = None
        self.undo_stack: List[Dict] = []
        # Allow a smaller editor window on compact displays
        self.setMinimumSize(480, 270)

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
            # finish polygon
            if len(self.current) >= 3:
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
            "current": copy.deepcopy(self.current)
        })

    def undo(self):
        if not self.undo_stack:
            return
        snap = self.undo_stack.pop()
        self.polygons = snap.get("polygons", [])
        self.current = snap.get("current", [])
        self.update()

    def clear_current(self):
        """Clear current in-progress polygon."""
        if self.current:
            self.push_undo()
        self.current = []
        self.update()

    def clear_all(self):
        """Clear all polygons and current drawing."""
        if self.polygons or self.current:
            self.push_undo()
        self.polygons = []
        self.current = []
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
        self.canvas.polygons = [
            [QPointF(p[0], p[1]) for p in zone.get("points", [])]
            for zone in self.zones.get(cam, [])
            if len(zone.get("points", [])) >= 3
        ]
        self.canvas.update()

    def init_ui(self):
        layout = QVBoxLayout()
        hl = QHBoxLayout()
        # Camera selector
        hl.addWidget(QLabel("Camera:"))
        self.camera_selector = QComboBox()
        self.reload_cameras()
        hl.addWidget(self.camera_selector)
        self.camera_selector.currentTextChanged.connect(self.on_camera_selected)
        
        load_btn = QPushButton("Load Snapshot")
        load_btn.clicked.connect(self.load_snapshot_from_selected)
        hl.addWidget(load_btn)

        undo_btn = QPushButton("↶ Undo")
        undo_btn.clicked.connect(self.canvas.undo)
        hl.addWidget(undo_btn)

        save_btn = QPushButton("Save Zones")
        save_btn.clicked.connect(self.save_zones)
        hl.addWidget(save_btn)

        loadzones_btn = QPushButton("Load Zones")
        loadzones_btn.clicked.connect(self.load_zones)
        hl.addWidget(loadzones_btn)

        layout.addLayout(hl)
        layout.addWidget(self.canvas)

        # Zone properties
        prop_h = QHBoxLayout()
        prop_h.addWidget(QLabel("Zone Name:"))
        self.zone_name = QLineEdit()
        prop_h.addWidget(self.zone_name)
        prop_h.addWidget(QLabel("Type:"))
        self.zone_type = QComboBox()
        self.zone_type.addItems(ZoneTypeChoices)
        prop_h.addWidget(self.zone_type)

        add_zone_btn = QPushButton("Create Zone from Current")
        add_zone_btn.clicked.connect(self.create_zone_from_current)
        prop_h.addWidget(add_zone_btn)

        clear_current_btn = QPushButton("Clear Current")
        clear_current_btn.clicked.connect(self.canvas.clear_current)
        prop_h.addWidget(clear_current_btn)

        clear_all_btn = QPushButton("Clear All")
        clear_all_btn.clicked.connect(self.canvas.clear_all)
        prop_h.addWidget(clear_all_btn)

        layout.addLayout(prop_h)

        # Zone list
        self.zone_list = QListWidget()
        self.zone_list.currentRowChanged.connect(self.on_zone_selected)
        layout.addWidget(self.zone_list)

        # Zone actions
        zone_actions = QHBoxLayout()
        del_zone_btn = QPushButton("Delete Selected Zone")
        del_zone_btn.clicked.connect(self.delete_selected_zone)
        zone_actions.addWidget(del_zone_btn)
        edit_zone_btn = QPushButton("Edit Selected Zone")
        edit_zone_btn.clicked.connect(self.edit_selected_zone)
        zone_actions.addWidget(edit_zone_btn)
        layout.addLayout(zone_actions)

        self.setLayout(layout)

    def reload_cameras(self):
        self.camera_selector.clear()
        cameras = self.config.get("cameras", {})
        for name in cameras.keys():
            self.camera_selector.addItem(name)

    def on_camera_selected(self, index):
        cam = self.camera_selector.currentText()
        if cam:
            self.current_camera = cam
            self.canvas.clear_all()  # clear previous camera drawing
            self.load_zones()  # auto load zones for selected camera

    def load_snapshot_from_selected(self):
        cam = self.camera_selector.currentText()
        if not cam:
            QMessageBox.warning(self, "Error", "No camera selected")
            return
        self.current_camera = cam
        cam_conf = self.config.get("cameras", {}).get(cam, {})
        url = cam_conf.get("rtsp_url")
        if not url:
            QMessageBox.warning(self, "Error", "RTSP URL not configured for this camera")
            return
        # try to read one frame using OpenCV
        try:
            cap = cv2.VideoCapture(url)
            ret, frame = cap.read()
            cap.release()
            if not ret:
                QMessageBox.critical(self, "Error", "Cannot capture frame from camera")
                return
            # convert to QImage
            h, w, ch = frame.shape
            bytes_per_line = ch * w
            qimg = QImage(frame.data, w, h, bytes_per_line, QImage.Format_BGR888)
            self.canvas.load_image(qimg)
            # Re-normalize existing zones in case source file used pixel coordinates.
            cam_zones = self.zones.get(self.current_camera, [])
            if cam_zones:
                for z in cam_zones:
                    z["points"] = self._normalized_points(z.get("points", []), image_size=(w, h))
                self._refresh_canvas_polygons()
            QMessageBox.information(self, "Snapshot", "Snapshot loaded")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Snapshot failed: {e}")

    def create_zone_from_current(self):
        name = self.zone_name.text().strip()
        if not name:
            QMessageBox.warning(self, "Error", "Zone name required")
            return
        pts_norm = self.canvas.current # These are already normalized QPointF
        if len(pts_norm) < 3:
            QMessageBox.warning(self, "Error", "Polygon must have at least 3 points")
            return
        # basic validation: points inside image rect
        w = self.canvas.width()
        h = self.canvas.height()
        for p in pts_norm:
            if not (0.0 <= p.x() <= 1.0 and 0.0 <= p.y() <= 1.0):
                QMessageBox.warning(self, "Error", "Polygon points must be inside image bounds")
                return
        z = {
            "name": name,
            "type": self.zone_type.currentText(),
            "points": [[p.x(), p.y()] for p in pts_norm]
        }
        if self.current_camera is None:
            QMessageBox.warning(self, "Error", "No camera loaded")
            return
        
        # Check if editing an existing zone or creating a new one
        selected_row = self.zone_list.currentRow()
        cam_zones = self.zones.setdefault(self.current_camera, [])
        can_edit = 0 <= selected_row < len(cam_zones)

        if can_edit:
            cam_zones[selected_row] = z
            item = self.zone_list.item(selected_row)
            if item:
                item.setText(f"{name} ({z['type']})")
            QMessageBox.information(self, "Updated", f"Zone '{name}' updated.")
        else:
            cam_zones.append(z)
            self.zone_list.addItem(QListWidgetItem(f"{name} ({z['type']})"))
            QMessageBox.information(self, "Created", f"Zone '{name}' created.")

        self.canvas.polygons = [
            [QPointF(p[0], p[1]) for p in zone['points']]
            for zone in cam_zones
        ]
        self.canvas.clear_current()
        self.canvas.update()
        # optional overlap check
        if self._check_overlap(self.current_camera, z):
            QMessageBox.warning(self, "Warning", "Zone overlaps existing zone")

    def save_zones(self):
        if not self.current_camera:
            QMessageBox.warning(self, "Error", "No camera selected/loaded")
            return
        zones = self.zones.get(self.current_camera, [])
        if not zones:
            QMessageBox.information(self, "Info", "No zones to save")
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
            QMessageBox.information(self, "Saved", f"Zones saved to {file_path}")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to save zones: {e}")

    def load_zones(self):
        cam = self.camera_selector.currentText()
        if not cam:
            QMessageBox.warning(self, "Error", "No camera selected")
            return
        zones_dir = Path(self.config.get("paths", {}).get("zones", "data/zones"))
        file_path = zones_dir / f"zones_{cam}.json"
        if not file_path.exists():
            QMessageBox.information(self, "Info", "No zones file for this camera")
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
            self.zone_list.clear()
            for z in normalized_loaded:
                self.zone_list.addItem(QListWidgetItem(f"{z.get('name','') } ({z.get('type','')})"))
            self._refresh_canvas_polygons()
            QMessageBox.information(self, "Loaded", f"Loaded {len(loaded)} zones for {cam}")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load zones: {e}")

    def delete_selected_zone(self):
        cam = self.current_camera or self.camera_selector.currentText()
        idx = self.zone_list.currentRow()
        if idx < 0:
            QMessageBox.information(self, "Info", "No zone selected")
            return
        if not cam or cam not in self.zones:
            QMessageBox.warning(self, "Error", "No zones loaded for this camera")
            return
        try:
            del self.zones[cam][idx]
            self.zone_list.takeItem(idx)
            self._refresh_canvas_polygons()
            QMessageBox.information(self, "Deleted", "Zone deleted")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to delete zone: {e}")

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
        self._refresh_canvas_polygons()
        self.canvas.update()

    def edit_selected_zone(self):
        """Prepare selected zone for editing by loading its points."""
        row = self.zone_list.currentRow()
        if row < 0:
            QMessageBox.information(self, "Info", "No zone selected")
            return
        self.on_zone_selected(row)
        QMessageBox.information(
            self,
            "Edit Zone",
            "Zone loaded to editor. Adjust points and click 'Create Zone from Current' to update."
        )

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
