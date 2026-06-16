"""PySide6 bridge to stream runtime analytics into the GUI layer."""

from __future__ import annotations

import logging
from typing import Dict

import numpy as np
from PySide6.QtCore import QObject, Signal

from services.camera_runtime import FrameAnalyticsPacket, MultiCameraRuntime


logger = logging.getLogger(__name__)


class SalonAIBridge(QObject):
    analytics_updated = Signal(dict)
    frame_ready = Signal(str, object)
    runtime_error = Signal(str)

    def __init__(self, runtime: MultiCameraRuntime) -> None:
        super().__init__()
        self.runtime = runtime
        self.runtime.subscribe(self._on_runtime_packet)

    def start(self) -> None:
        try:
            self.runtime.start()
        except Exception as exc:
            msg = f"Failed to start salon runtime: {exc}"
            logger.exception(msg)
            self.runtime_error.emit(msg)

    def stop(self) -> None:
        try:
            self.runtime.stop()
        except Exception as exc:
            msg = f"Failed to stop salon runtime: {exc}"
            logger.exception(msg)
            self.runtime_error.emit(msg)

    def latest_status(self) -> Dict[str, Dict]:
        return self.runtime.latest_packets()

    def _on_runtime_packet(self, packet: FrameAnalyticsPacket, frame: np.ndarray) -> None:
        self.analytics_updated.emit(packet.to_dict())
        self.frame_ready.emit(packet.camera_id, frame)
