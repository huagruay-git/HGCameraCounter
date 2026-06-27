"""Modern theme + sidebar shell for the HG Camera Counter controller.

Brand scheme: yellow + white + black. Light/white content surfaces, a black
sidebar + status bar, and brand-yellow accents (active nav, hero, highlights).

Provides:
- THEME_QSS: the application stylesheet.
- SidebarTabWidget: a drop-in stand-in for QTabWidget that exposes addTab() but
  renders the pages as a left-hand sidebar nav + stacked content area.
- apply_theme(app): apply the stylesheet to a QApplication.
- Color constants reused by the dashboard cards.
"""
from __future__ import annotations

import re

from PySide6.QtCore import QSize, Qt
from PySide6.QtWidgets import (
    QApplication,
    QHBoxLayout,
    QListWidget,
    QListWidgetItem,
    QStackedWidget,
    QWidget,
)

# --- Brand palette (yellow / white / black) ---
ACCENT = "#F5C518"        # brand yellow
ACCENT_HOVER = "#FFE066"
ACCENT_TEXT = "#161616"   # black text on yellow
BG = "#F6F6F3"            # app/content background (light)
SIDEBAR_BG = "#161616"    # black sidebar
SURFACE = "#FFFFFF"       # white cards
BORDER = "#E3E3DD"
TEXT = "#1A1A1A"          # near-black text
TEXT_MUTED = "#6B6B64"
OK = "#3B6D11"            # occupied / running (green)
WARN = "#8A5412"          # amber-dark (text on yellow tints)
DANGER = "#A32D2D"        # offline / stopped (red)

THEME_QSS = f"""
* {{ font-family: "Segoe UI", "Inter", "Tahoma", sans-serif; }}
QWidget {{ font-size: 13px; color: {TEXT}; }}
QMainWindow, QMainWindow > QWidget {{ background: {BG}; }}
QToolTip {{ background: {SIDEBAR_BG}; color: {ACCENT}; border: none; padding: 6px 8px; border-radius: 6px; }}

QListWidget#NavList {{
    background: {SIDEBAR_BG};
    border: none;
    outline: 0;
    padding: 10px 0;
}}
QListWidget#NavList::item {{
    color: #C8C8C2;
    padding: 10px 16px;
    margin: 2px 10px;
    border-radius: 8px;
}}
QListWidget#NavList::item:hover {{ background: #2A2A2A; color: #FFFFFF; }}
QListWidget#NavList::item:selected {{ background: {ACCENT}; color: {ACCENT_TEXT}; }}

QStackedWidget#ContentStack {{ background: {BG}; }}

QGroupBox {{
    background: {SURFACE};
    border: 1px solid {BORDER};
    border-radius: 12px;
    margin-top: 16px;
    padding: 12px;
    color: {TEXT};
    font-weight: 500;
}}
QGroupBox::title {{ subcontrol-origin: margin; left: 12px; padding: 0 5px; color: {TEXT}; }}

QPushButton {{
    background: {SURFACE};
    border: 1px solid #D7D7D0;
    border-radius: 8px;
    padding: 7px 14px;
    color: {TEXT};
}}
QPushButton:hover {{ background: #FFF8E0; border-color: {ACCENT}; }}
QPushButton:pressed {{ background: #F5EFD8; }}
QPushButton:disabled {{ color: #ABABA4; background: #F1F1ED; border-color: {BORDER}; }}

QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox, QPlainTextEdit, QTextEdit {{
    background: {SURFACE};
    border: 1px solid #D7D7D0;
    border-radius: 8px;
    padding: 6px 8px;
    color: {TEXT};
    selection-background-color: {ACCENT};
    selection-color: {ACCENT_TEXT};
}}
QLineEdit:focus, QComboBox:focus, QSpinBox:focus, QDoubleSpinBox:focus,
QPlainTextEdit:focus, QTextEdit:focus {{ border: 1px solid {ACCENT}; }}
QComboBox::drop-down {{ border: none; width: 22px; }}
QComboBox QAbstractItemView {{ background: {SURFACE}; color: {TEXT}; selection-background-color: {ACCENT}; selection-color: {ACCENT_TEXT}; }}

QTabWidget::pane {{ border: 1px solid {BORDER}; border-radius: 8px; top: -1px; }}
QTabBar::tab {{ background: transparent; padding: 8px 14px; color: {TEXT_MUTED}; }}
QTabBar::tab:selected {{ color: {TEXT}; border-bottom: 2px solid {ACCENT}; }}

QScrollBar:vertical {{ background: transparent; width: 10px; margin: 0; }}
QScrollBar::handle:vertical {{ background: #C9C9C2; border-radius: 5px; min-height: 26px; }}
QScrollBar::handle:vertical:hover {{ background: {ACCENT}; }}
QScrollBar:horizontal {{ background: transparent; height: 10px; margin: 0; }}
QScrollBar::handle:horizontal {{ background: #C9C9C2; border-radius: 5px; min-width: 26px; }}
QScrollBar::handle:horizontal:hover {{ background: {ACCENT}; }}
QScrollBar::add-line, QScrollBar::sub-line {{ width: 0; height: 0; }}
QScrollBar::add-page, QScrollBar::sub-page {{ background: transparent; }}

QStatusBar {{ background: {SIDEBAR_BG}; color: #C8C8C2; border: none; }}
QStatusBar QLabel {{ color: #C8C8C2; }}
QLabel {{ background: transparent; }}
QTreeWidget, QTableWidget, QTableView {{
    background: {SURFACE}; color: {TEXT};
    border: 1px solid {BORDER}; border-radius: 8px;
    gridline-color: #EEEEE8; alternate-background-color: #FAFAF7;
}}
QTreeWidget::item:selected, QTableWidget::item:selected, QTableView::item:selected {{ background: {ACCENT}; color: {ACCENT_TEXT}; }}
QHeaderView::section {{ background: {BG}; color: {TEXT_MUTED}; border: none; border-bottom: 1px solid {BORDER}; padding: 6px 8px; }}
QCheckBox {{ color: {TEXT}; }}
QCheckBox::indicator {{ width: 16px; height: 16px; border: 1px solid #C4C4BC; border-radius: 4px; background: {SURFACE}; }}
QCheckBox::indicator:checked {{ background: {ACCENT}; border-color: {ACCENT}; }}
QProgressBar {{ background: #EFEFEA; border: 1px solid {BORDER}; border-radius: 6px; text-align: center; color: {TEXT}; }}
QProgressBar::chunk {{ background: {ACCENT}; border-radius: 6px; }}
"""

_PREFIX_JUNK = re.compile(r"^[^0-9A-Za-z฀-๿]+")


def _clean_label(text: str) -> str:
    raw = str(text or "").strip()
    cleaned = _PREFIX_JUNK.sub("", raw).strip()
    return cleaned or raw


class SidebarTabWidget(QWidget):
    """QTabWidget-compatible (addTab) widget rendered as sidebar nav + stacked pages."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._nav = QListWidget()
        self._nav.setObjectName("NavList")
        self._nav.setFixedWidth(198)
        self._nav.setFocusPolicy(Qt.NoFocus)
        self._nav.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        self._stack = QStackedWidget()
        self._stack.setObjectName("ContentStack")

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(self._nav)
        layout.addWidget(self._stack, 1)
        # Don't let the (tall) page content force a large minimum window height —
        # allow the window to be shrunk; content clips/scrolls. Matches the old
        # FlexibleTabWidget behavior.
        layout.setSizeConstraint(QHBoxLayout.SizeConstraint.SetNoConstraint)
        self._stack.setMinimumHeight(0)

        self._nav.currentRowChanged.connect(self._stack.setCurrentIndex)

    def minimumSizeHint(self) -> QSize:
        return QSize(640, 360)

    # --- QTabWidget-compatible API (only what main.py uses) ---
    def addTab(self, widget: QWidget, label: str) -> int:
        index = self._stack.addWidget(widget)
        item = QListWidgetItem(_clean_label(label))
        item.setSizeHint(QSize(0, 42))
        self._nav.addItem(item)
        if self._nav.currentRow() < 0:
            self._nav.setCurrentRow(0)
        return index

    def setCurrentIndex(self, index: int) -> None:
        self._nav.setCurrentRow(index)

    def currentIndex(self) -> int:
        return self._nav.currentRow()

    def count(self) -> int:
        return self._stack.count()

    def widget(self, index: int) -> QWidget:
        return self._stack.widget(index)

    def setTabText(self, index: int, text: str) -> None:
        item = self._nav.item(index)
        if item is not None:
            item.setText(_clean_label(text))


def apply_theme(app: QApplication) -> None:
    """Apply the flat yellow/white/black brand stylesheet to the application."""
    app.setStyleSheet(THEME_QSS)
