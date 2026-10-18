# -*- coding: utf-8 -*-
"""
深色主题（Fusion + 深色 QPalette + 控件 QSS）。

并针对 DS-Labeler 的三面板（频谱 / 波形 / 钢琴卷帘）布局补充分隔条样式，
消除面板间"黑白割裂"的观感。
"""
from __future__ import annotations

# 统一配色常量（视图自绘也引用，保证同一套色）
BG = "#1e2026"          # 窗口底
PANEL = "#262830"       # 面板底
BASE = "#181a20"        # 输入/列表底
TEXT = "#d2d6dc"
TEXT_DIM = "#9aa0ab"
LINE = "#33363f"        # 分割线 / 边框
ACCENT = "#6aa0ff"      # 强调
SELECT = "#3a6ec8"      # 选中

# 视图内部用色
CANVAS_BG = "#181a20"       # 频谱底
WAVE_BG = "#161820"         # 波形底
ROLL_BG = "#14161c"         # 钢琴卷帘底
GRID = "#2a2d36"            # 网格线
RULER_BG = "#20222a"        # 顶部刻度带
LABEL_BG = "#1b1d24"        # 音素标签带


def apply_theme(app) -> None:
    """把深色主题应用到 QApplication（Fusion + QPalette + QSS）。"""
    app.setStyle("Fusion")
    from PySide6.QtCore import Qt
    from PySide6.QtGui import QPalette, QColor

    bg = QColor(30, 32, 38)
    panel = QColor(38, 40, 46)
    text = QColor(210, 214, 220)
    sel = QColor(58, 110, 200)

    pal = QPalette()
    pal.setColor(QPalette.Window, bg)
    pal.setColor(QPalette.WindowText, text)
    pal.setColor(QPalette.Base, QColor(24, 26, 32))
    pal.setColor(QPalette.AlternateBase, panel)
    pal.setColor(QPalette.ToolTipBase, panel)
    pal.setColor(QPalette.ToolTipText, text)
    pal.setColor(QPalette.Text, text)
    pal.setColor(QPalette.Button, panel)
    pal.setColor(QPalette.ButtonText, text)
    pal.setColor(QPalette.BrightText, Qt.red)
    pal.setColor(QPalette.Link, QColor(90, 160, 255))
    pal.setColor(QPalette.Highlight, sel)
    pal.setColor(QPalette.HighlightedText, Qt.white)
    pal.setColor(QPalette.PlaceholderText, QColor(120, 125, 135))
    app.setPalette(pal)

    app.setStyleSheet(_QSS)


_QSS = """
QWidget { font-family: "Microsoft YaHei", "Segoe UI"; font-size: 12px; }
QMainWindow, QDialog { background: #1e2026; }

QMenuBar { background: #20222a; color: #d2d6dc; border-bottom: 1px solid #2c2f38; }
QMenuBar::item { padding: 4px 10px; background: transparent; }
QMenuBar::item:selected { background: #2c2f3a; }
QMenu { background: #262933; color: #d2d6dc; border: 1px solid #3a3e4a; padding: 4px; }
QMenu::item { padding: 4px 22px 4px 14px; }
QMenu::item:selected { background: #2c2f3a; }
QMenu::separator { height: 1px; background: #3a3e4a; margin: 4px 6px; }

QStatusBar { background: #20222a; color: #b8bdc6; }
QStatusBar::item { border: none; }

QToolBar { background: #20222a; spacing: 3px; padding: 3px; border-bottom: 1px solid #2c2f38; }
QToolBar::separator { background: #3a3e4a; width: 1px; margin: 3px 5px; }
QToolButton { background: #333641; color: #e2e5ea; border: 1px solid #454a57;
              padding: 3px 9px; border-radius: 3px; }
QToolButton:hover { background: #3c404d; }
QToolButton:pressed { background: #2a2d36; }
QToolButton:checked { background: #3a6ec8; color: #ffffff; border-color: #5b8fe0; }
QToolButton:disabled { color: #7a7f8a; }

QListWidget { background: #181a20; color: #d2d6dc; border: 1px solid #33363f; outline: none; }
QListWidget::item { padding: 3px 6px; }
QListWidget::item:selected { background: #3a6ec8; color: #ffffff; }
QListWidget::item:hover { background: #2c2f3a; }

QLabel { color: #cfd3da; }
QLabel#tierBar { background: #20222a; color: #d2d6dc; padding: 4px 8px;
                 border-bottom: 1px solid #2c2f38; }
QLabel#tierName { color: #6aa0ff; font-weight: bold; }

QPushButton { background: #333641; color: #e2e5ea; border: 1px solid #454a57;
              padding: 3px 10px; border-radius: 3px; }
QPushButton:hover { background: #3c404d; }
QPushButton:pressed { background: #2a2d36; }
QPushButton:disabled { color: #7a7f8a; }

QLineEdit, QSpinBox, QDoubleSpinBox { background: #181a20; color: #d2d6dc;
                                      border: 1px solid #454a57; padding: 2px 5px; }

QComboBox { background: #262933; color: #d2d6dc; border: 1px solid #454a57; padding: 2px 6px; }
QComboBox QAbstractItemView { background: #262933; color: #d2d6dc;
                              selection-background-color: #2c2f3a; border: 1px solid #3a3e4a; }

QSlider::groove:horizontal { height: 5px; background: #333641; border-radius: 2px; }
QSlider::handle:horizontal { width: 12px; margin: -4px 0; border-radius: 6px; background: #6aa0ff; }
QSlider::sub-page:horizontal { background: #4a7ad0; border-radius: 2px; }

QToolTip { background: #262933; color: #e2e5ea; border: 1px solid #454a57; }

QSplitter::handle { background: #14161a; }
QSplitter::handle:horizontal { width: 4px; }
QSplitter::handle:vertical { height: 4px; }
QSplitter::handle:hover { background: #3a6ec8; }

QScrollBar:vertical { background: #1b1d24; width: 10px; margin: 0; }
QScrollBar::handle:vertical { background: #3a3e4a; border-radius: 4px; min-height: 24px; }
QScrollBar::handle:vertical:hover { background: #4a4f5e; }
QScrollBar:horizontal { background: #1b1d24; height: 10px; margin: 0; }
QScrollBar::handle:horizontal { background: #3a3e4a; border-radius: 4px; min-width: 24px; }
QScrollBar::add-line, QScrollBar::sub-line { width: 0; height: 0; }
"""
