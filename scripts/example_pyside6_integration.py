#!/usr/bin/env python3
"""Minimal PySide6 example that plugs SalonAIBridge into a GUI."""

from __future__ import annotations

import sys
from pathlib import Path

from PySide6.QtWidgets import QApplication, QLabel, QPushButton, QVBoxLayout, QWidget


PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from app.salon_ai_system import SalonAISystem
from ui.salon_ai_bridge import SalonAIBridge


class DemoWindow(QWidget):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Salon AI Runtime Demo")
        self.setMinimumWidth(520)

        self.status_label = QLabel("Runtime: idle")
        self.count_label = QLabel("Haircuts=0 | Washes=0 | Staff=0")
        self.toggle_btn = QPushButton("Start Runtime")

        layout = QVBoxLayout(self)
        layout.addWidget(self.status_label)
        layout.addWidget(self.count_label)
        layout.addWidget(self.toggle_btn)

        self.system = SalonAISystem("data/config/salon_ai.runtime.yaml")
        self.bridge = SalonAIBridge(self.system.runtime)
        self.bridge.analytics_updated.connect(self._on_analytics)
        self.bridge.runtime_error.connect(self._on_error)

        self.running = False
        self.toggle_btn.clicked.connect(self.toggle_runtime)

    def toggle_runtime(self) -> None:
        if not self.running:
            self.bridge.start()
            self.running = True
            self.status_label.setText("Runtime: running")
            self.toggle_btn.setText("Stop Runtime")
            return
        self.bridge.stop()
        self.running = False
        self.status_label.setText("Runtime: stopped")
        self.toggle_btn.setText("Start Runtime")

    def _on_analytics(self, payload: dict) -> None:
        counters = payload.get("counters", {})
        haircuts = int(counters.get("haircuts_total", 0))
        washes = int(counters.get("washes_total", 0))
        staff = int(counters.get("staff_seen_total", 0))
        self.count_label.setText(f"Haircuts={haircuts} | Washes={washes} | Staff={staff}")

    def _on_error(self, msg: str) -> None:
        self.status_label.setText(f"Runtime error: {msg}")


def main() -> int:
    app = QApplication(sys.argv)
    win = DemoWindow()
    win.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
