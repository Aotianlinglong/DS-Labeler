# -*- coding: utf-8 -*-
"""
统一设置面板（左导航 + 右详情）：

  左栏          右栏详情
  ├ 语言    →   主语言（**跟随工程**）+ 全局无声音素（自定义词典/元辅音/音素系统已移除）
  ├ 频谱    →   窗函数 / 频率轴(mel·线性·对数) / FFT 间隔 / 最大频率 / 分辨率 / dB / 配色
  ├ 波形    →   钢琴窗波形的上下最大宽度 / 每段外扩时长 / PIT 音高线粗细
  ├ 面板    →   四个面板（波形 / 钢琴卷帘 / 口型 / 频谱）的上下顺序 + 各自显隐 + 口型纵轴
  └ 配色    →   音符 / PIT / 两处波形 / 口型两色的颜色 + 配色方案（内置 + 自存预设）

**归属**：
  · 主语言与全局无声音素 → **工程**（随 `project.save()` 落 ds_labeler_settings.json）；
  · 频谱 / 波形 / 面板 / 配色 → **软件级**（`ProjectSettings`，写软件根目录 app_settings.json，
    所有工程共用，切换工程不丢）。
  运行态刷新由主窗口负责（`_refresh_language_state` / `_apply_panes` / `_apply_wave_metrics` /
  `_apply_colors` / 重算当前句频谱）。
窗长按采样率自适应，故意不暴露（谐波不被主瓣糊掉的前提）。
"""
from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QAbstractItemView, QCheckBox, QColorDialog, QComboBox, QDialog, QDialogButtonBox,
    QDoubleSpinBox, QFormLayout, QGridLayout, QHBoxLayout, QInputDialog, QLabel,
    QLineEdit, QListWidget, QListWidgetItem, QMessageBox, QPushButton, QSlider,
    QSpinBox, QStackedWidget, QVBoxLayout, QWidget,
)

from core.phonemes import DEFAULT_GLOBAL_PHONEMES, LANGS
from core.project import (
    COLOR_KEYS, COLOR_SCHEMES, DEFAULT_COLORS, Project, ProjectSettings, scheme_of,
)
from core.spectrogram import (
    CMAP_LABELS, CMAPS, DEFAULT_CMAP, SCALE_LABELS, SCALES, WINDOW_NAMES, WINDOWS,
)

LANG_NAMES = {"zh": "中文 zh", "ja": "日语 ja", "en": "英语 en", "yue": "粤语 yue"}


class LangPage(QWidget):
    """语言页：**只留「主语言」与「全局无声音素」**（其余跟工程走/用内置默认）。

    自定义词典、元辅音表、流音表、符号类型表、音素系统这些「后面再设置是无用功」
    （只在跑 GAME 时才会用到）——故从界面移除；内核仍按内置默认（zh/ja 二分式、yue 单音节、
    en 多音节，见 core.phonemes）解析，导出的 ph_num 逻辑不受影响。
    主语言是**工程级**设置（不同数据集语言不同），故没有工程时本页不可用。
    """

    def __init__(self, project: Project | None, parent=None):
        super().__init__(parent)
        self.project = project
        root = QVBoxLayout(self)
        grid = QGridLayout()

        grid.addWidget(QLabel("主语言"), 0, 0)
        self.cb_main = QComboBox()
        for lang in LANGS:
            self.cb_main.addItem(LANG_NAMES.get(lang, lang), lang)
        if project is not None and project.main_lang in LANGS:
            self.cb_main.setCurrentIndex(list(LANGS).index(project.main_lang))
        grid.addWidget(self.cb_main, 0, 1)
        grid.setColumnStretch(1, 1)
        root.addLayout(grid)

        h = QHBoxLayout()
        h.addWidget(QLabel("全局无声音素（空格分隔）"))
        self.ed_global = QLineEdit(
            " ".join(project.global_phonemes if project is not None
                     else list(DEFAULT_GLOBAL_PHONEMES)))
        h.addWidget(self.ed_global, 1)
        root.addLayout(h)

        tip = QLabel(
            "主语言随**工程**保存（不同数据集语言不同）；混搭语言句按工程主语言走推理。\n"
            "音素系统与元辅/流音/词典一律用**内置默认**（zh/ja 二分式、yue 单音节、en 多音节）"
            "——这些只影响 GAME 推理侧，界面上再设一遍是无用功，故已从界面移除。\n"
            "全局无声音素影响显示语义（SP/AP 等照常渲染），可按需增删。")
        tip.setWordWrap(True)
        root.addWidget(tip)
        if project is None:
            self.cb_main.setEnabled(False)
            self.ed_global.setEnabled(False)
            warn = QLabel("⚠ 未打开工程：主语言随工程保存，打开/新建工程后才可修改。")
            warn.setWordWrap(True)
            root.addWidget(warn)
        root.addStretch(1)

    def apply_to_project(self, project: Project | None = None) -> None:
        p = project or self.project
        if p is None:
            return
        p.main_lang = self.cb_main.currentData() or p.main_lang
        p.global_phonemes = self.ed_global.text().split() or list(DEFAULT_GLOBAL_PHONEMES)
        p.save()


class SpecPage(QWidget):
    """频谱显示（原「频谱设置」页）：窗函数 / 间隔 / 频率范围 / dB / 配色。"""

    def __init__(self, settings: ProjectSettings, parent=None):
        super().__init__(parent)
        st = settings
        root = QVBoxLayout(self)
        form = QFormLayout()

        self.cb_window = QComboBox()
        for w in WINDOWS:
            self.cb_window.addItem(WINDOW_NAMES.get(w, w), w)
        _w = st.spec_window if st.spec_window in WINDOWS else WINDOWS[0]
        self.cb_window.setCurrentIndex(list(WINDOWS).index(_w))
        form.addRow("窗函数", self.cb_window)

        self.cb_scale = QComboBox()
        for s in SCALES:
            self.cb_scale.addItem(SCALE_LABELS.get(s, s), s)
        _s = st.spec_scale if st.spec_scale in SCALES else SCALES[0]
        self.cb_scale.setCurrentIndex(list(SCALES).index(_s))
        self.cb_scale.setToolTip(
            "行 → 频率 的映射方式：mel（感知均匀）／线性 Hz（谐波等间距）／对数 Hz。"
            "行数不受它影响，仍由下面的「频率分辨率」决定")
        form.addRow("频率轴", self.cb_scale)

        self.sp_hop = QSpinBox()
        self.sp_hop.setRange(16, 4096)
        self.sp_hop.setValue(int(st.spec_hop))
        self.sp_hop.setSuffix("  采样点/帧")
        form.addRow("FFT 间隔", self.sp_hop)

        self.sp_fmax = QDoubleSpinBox()
        self.sp_fmax.setRange(500.0, 96000.0)
        self.sp_fmax.setDecimals(0)
        self.sp_fmax.setSingleStep(500.0)
        self.sp_fmax.setValue(float(st.spec_max_freq))
        self.sp_fmax.setSuffix("  Hz")
        form.addRow("最大频率", self.sp_fmax)

        self.sp_mel = QDoubleSpinBox()
        self.sp_mel.setRange(1.0, 200.0)
        self.sp_mel.setDecimals(1)
        self.sp_mel.setSingleStep(1.0)
        self.sp_mel.setValue(float(st.spec_mel_res))
        self.sp_mel.setSuffix("  mel/行")
        form.addRow("频率分辨率", self.sp_mel)

        self.sp_dbmin = QDoubleSpinBox()
        self.sp_dbmin.setRange(-160.0, 160.0)
        self.sp_dbmin.setDecimals(1)
        self.sp_dbmin.setValue(float(st.spec_db_min))
        self.sp_dbmin.setSuffix("  dB")
        form.addRow("最小强度", self.sp_dbmin)

        self.sp_dbmax = QDoubleSpinBox()
        self.sp_dbmax.setRange(-160.0, 240.0)
        self.sp_dbmax.setDecimals(1)
        self.sp_dbmax.setValue(float(st.spec_db_max))
        self.sp_dbmax.setSuffix("  dB")
        form.addRow("最大强度", self.sp_dbmax)

        self.cb_cmap = QComboBox()
        for name in CMAPS:
            self.cb_cmap.addItem(CMAP_LABELS.get(name, name), name)
        _c = st.spectrum_preset if st.spectrum_preset in CMAPS else DEFAULT_CMAP
        self.cb_cmap.setCurrentIndex(list(CMAPS).index(_c))
        form.addRow("配色", self.cb_cmap)

        root.addLayout(form)
        tip = QLabel("窗长按采样率自适应（44.1kHz → 4096），以保证谐波不被主瓣糊掉，故不在此暴露。")
        tip.setWordWrap(True)
        root.addWidget(tip)
        root.addStretch(1)

    def values(self) -> ProjectSettings:
        return ProjectSettings(
            spec_window=self.cb_window.currentData() or WINDOWS[0],
            spec_scale=self.cb_scale.currentData() or SCALES[0],
            spec_hop=int(self.sp_hop.value()),
            spec_max_freq=float(self.sp_fmax.value()),
            spec_mel_res=float(self.sp_mel.value()),
            spec_db_min=float(self.sp_dbmin.value()),
            spec_db_max=float(self.sp_dbmax.value()),
            spectrum_preset=self.cb_cmap.currentData() or DEFAULT_CMAP,
        )

    def apply_to_settings(self, settings: ProjectSettings) -> None:
        v = self.values()
        settings.spec_window = v.spec_window
        settings.spec_scale = v.spec_scale      # 频率轴：切了要重算频谱（主窗负责）
        settings.spec_hop = v.spec_hop
        settings.spec_max_freq = v.spec_max_freq
        settings.spec_mel_res = v.spec_mel_res
        settings.spec_db_min = v.spec_db_min
        settings.spec_db_max = v.spec_db_max
        settings.spectrum_preset = v.spectrum_preset


class WavePage(QWidget):
    """钢琴窗波形预览：上下最大宽度 + 每段外扩时长 + PIT 线粗细。"""

    def __init__(self, settings: ProjectSettings, parent=None):
        super().__init__(parent)
        st = settings
        root = QVBoxLayout(self)
        form = QFormLayout()

        self.sp_h = QSpinBox()
        self.sp_h.setRange(8, 400)
        self.sp_h.setValue(int(round(st.wave_h)))
        self.sp_h.setSuffix("  px")
        form.addRow("波形上下最大宽度", self.sp_h)

        self.sp_ext = QDoubleSpinBox()
        self.sp_ext.setRange(0.0, 2000.0)
        self.sp_ext.setDecimals(0)
        self.sp_ext.setSingleStep(10.0)
        self.sp_ext.setValue(float(st.wave_ext_ms))
        self.sp_ext.setSuffix("  ms")
        form.addRow("每段外扩时长", self.sp_ext)

        # PIT（F0）音高线粗细：最细 1.0px（现状）、最粗 2.0px（更早那版）
        self.sl_pit = QSlider(Qt.Horizontal)
        self.sl_pit.setRange(10, 20)              # 10 → 1.0px，20 → 2.0px（0.1px 一档）
        self.sl_pit.setValue(int(round(float(st.pit_width) * 10)))
        self.lb_pit = QLabel(f"{self.sl_pit.value() / 10.0:.1f} px")
        self.lb_pit.setFixedWidth(48)
        self.sl_pit.valueChanged.connect(
            lambda v: self.lb_pit.setText(f"{v / 10.0:.1f} px"))
        pit_row = QWidget()
        pr = QHBoxLayout(pit_row)
        pr.setContentsMargins(0, 0, 0, 0)
        pr.addWidget(self.sl_pit, 1)
        pr.addWidget(self.lb_pit, 0)
        form.addRow("PIT 音高线粗细", pit_row)

        root.addLayout(form)
        tip = QLabel("钢琴卷帘里每个音符组下方的波形。\n"
                     "上下最大宽度用**像素**：纵向不随行高 / 缩放变，放大后波形不会跟着变粗。\n"
                     "外扩用**时长**而不是像素：像素外扩会随时间轴缩放变长变短，"
                     "固定时长才能在任意缩放下都稳定地「多显示一截」，方便对着波形改音素边界。\n"
                     "PIT 线越细越不挡波形与音素边界，越粗越醒目。")
        tip.setWordWrap(True)
        root.addWidget(tip)
        root.addStretch(1)

    def values(self) -> tuple[float, float, float]:
        return (float(self.sp_h.value()), float(self.sp_ext.value()),
                self.sl_pit.value() / 10.0)

    def apply_to_settings(self, settings: ProjectSettings) -> None:
        h, ext, pit = self.values()
        settings.wave_h = h
        settings.wave_ext_ms = ext
        settings.pit_width = pit


class PanelPage(QWidget):
    """面板（波形 / 钢琴卷帘 / 口型 / 频谱）的上下顺序 + 各自显隐 + 标签字号。"""

    LABELS = {"wave": "波形面板", "roll": "钢琴卷帘", "curve": "口型面板", "spec": "频谱"}

    def __init__(self, settings: ProjectSettings, parent=None):
        super().__init__(parent)
        st = settings
        root = QVBoxLayout(self)
        root.addWidget(QLabel("上下顺序（由上到下）"))
        self.lst = QListWidget()
        self.lst.setSelectionMode(QAbstractItemView.SingleSelection)
        self.lst.setMaximumHeight(120)
        for k in st.panel_order:
            it = QListWidgetItem(self.LABELS.get(k, k))
            it.setData(Qt.UserRole, k)
            self.lst.addItem(it)
        self.lst.setCurrentRow(0)
        root.addWidget(self.lst)

        row = QHBoxLayout()
        self.btn_up = QPushButton("上移")
        self.btn_dn = QPushButton("下移")
        self.btn_up.clicked.connect(lambda: self._move(-1))
        self.btn_dn.clicked.connect(lambda: self._move(1))
        row.addWidget(self.btn_up)
        row.addWidget(self.btn_dn)
        row.addStretch(1)
        root.addLayout(row)

        self.chk_wave = QCheckBox("显示独立波形面板（关掉后波形只在钢琴卷帘里显示）")
        self.chk_wave.setChecked(bool(st.wave_panel))
        root.addWidget(self.chk_wave)
        self.chk_roll = QCheckBox("显示钢琴卷帘")
        self.chk_roll.setChecked(bool(st.roll_panel))
        root.addWidget(self.chk_roll)
        self.chk_curve = QCheckBox("显示口型面板（R3MOE 开口度曲线；需自备模型权重）")
        self.chk_curve.setChecked(bool(st.curve_panel))
        root.addWidget(self.chk_curve)
        self.chk_spec = QCheckBox("显示频谱窗")
        self.chk_spec.setChecked(bool(st.spec_panel))
        root.addWidget(self.chk_spec)

        self.chk_fit = QCheckBox("口型曲线按当前句峰值自适应高度"
                                 "（把该句最大张口顶到面板顶端；关掉＝绝对标尺 0…1）")
        self.chk_fit.setChecked(bool(st.curve_autoscale))
        root.addWidget(self.chk_fit)

        frow = QHBoxLayout()
        frow.addWidget(QLabel("标签字号"))
        self.sp_font = QSpinBox()
        self.sp_font.setRange(5, 16)
        self.sp_font.setSuffix(" pt")
        self.sp_font.setValue(int(st.label_font_size))
        frow.addWidget(self.sp_font)
        frow.addWidget(QLabel("（音符块内 / 词首标签 / 音素名共用）"))
        frow.addStretch(1)
        root.addLayout(frow)

        self.chk_inner = QCheckBox("在音符块内显示标签文字（默认关：PIT 曲线会盖住块内文字）")
        self.chk_inner.setChecked(bool(st.note_label_inner))
        root.addWidget(self.chk_inner)

        tip = QLabel("顺序立即生效：切面板只是把面板按新顺序重新挂进分隔条，"
                     "不会重载频谱 / 音符 / 播放头，时间轴也保持不变。\n"
                     "各面板都可关闭（隐藏后自动从顺序里跳过，与「显示」菜单同一份设置）。\n"
                     "口型面板默认关闭；打开后会按「切到该句时计算并缓存」，"
                     "也可用菜单「工具 → 计算所有文件口型」一次性算完全部句。\n"
                     "「按峰值自适应」＝纵轴满量程取当前句开口度最大值（面板左侧标出峰值），"
                     "关掉则按绝对标尺 0（闭嘴）… 1（最大张口）——跨句比较张口幅度时用后者；"
                     "该项也可在「显示」菜单里随手切换（同一份设置）。\n"
                     "音素标签始终写在词首音符的块外上方，所以关掉块内文字不丢信息。")
        tip.setWordWrap(True)
        root.addWidget(tip)
        root.addStretch(1)

    def _move(self, d: int):
        i = self.lst.currentRow()
        j = i + d
        if i < 0 or not (0 <= j < self.lst.count()):
            return
        it = self.lst.takeItem(i)
        self.lst.insertItem(j, it)
        self.lst.setCurrentRow(j)

    def order(self) -> list[str]:
        return [self.lst.item(i).data(Qt.UserRole) for i in range(self.lst.count())]

    def apply_to_settings(self, settings: ProjectSettings) -> None:
        settings.panel_order = self.order()
        settings.wave_panel = bool(self.chk_wave.isChecked())
        settings.roll_panel = bool(self.chk_roll.isChecked())
        settings.curve_panel = bool(self.chk_curve.isChecked())
        settings.spec_panel = bool(self.chk_spec.isChecked())
        settings.curve_autoscale = bool(self.chk_fit.isChecked())
        settings.label_font_size = int(self.sp_font.value())
        settings.note_label_inner = bool(self.chk_inner.isChecked())


class PalettePage(QWidget):
    """配色：音符 / PIT / 两处波形 / 口型曲线的颜色 + 配色方案（内置 + 用户自存预设）。"""

    ROWS = (("note_color", "音符颜色"),
            ("pit_color", "PIT 音高线颜色"),
            ("wave_panel_color", "波形面板波形颜色"),
            ("wave_roll_color", "钢琴卷帘波形颜色"),
            ("curve_line_color", "口型曲线描边颜色"),
            ("curve_fill_color", "口型曲线填充颜色"))

    def __init__(self, settings: ProjectSettings, parent=None):
        super().__init__(parent)
        st = settings
        # 用户自存预设（内存态；apply_to_project 时才落盘）——与内置 COLOR_SCHEMES 分开
        self.presets: dict[str, dict[str, str]] = {k: dict(v)
                                                  for k, v in (st.color_presets or {}).items()}
        root = QVBoxLayout(self)
        form = QFormLayout()

        self.cb_scheme = QComboBox()
        self.cb_scheme.currentIndexChanged.connect(self._on_scheme)
        row_sch = QHBoxLayout()
        row_sch.setContentsMargins(0, 0, 0, 0)
        row_sch.setSpacing(6)
        row_sch.addWidget(self.cb_scheme, 1)
        self.btn_save = QPushButton("存为预设…")
        self.btn_save.clicked.connect(self._save_preset)
        row_sch.addWidget(self.btn_save, 0)
        self.btn_del = QPushButton("删除预设")
        self.btn_del.clicked.connect(self._del_preset)
        row_sch.addWidget(self.btn_del, 0)
        w_sch = QWidget()
        w_sch.setLayout(row_sch)
        form.addRow("配色方案", w_sch)
        self._rebuild_schemes()               # 按钮先建好再填下拉（_rebuild 会用到 btn_del）

        self.edits: dict[str, QLineEdit] = {}
        self.swatches: dict[str, QPushButton] = {}
        for key, title in self.ROWS:
            ed = QLineEdit(str(getattr(st, key, "") or DEFAULT_COLORS[key]))
            ed.setFixedWidth(90)
            ed.editingFinished.connect(self._mark_scheme)
            sw = QPushButton()
            sw.setFixedSize(36, 20)
            sw.setToolTip("点这里取色")
            sw.clicked.connect(lambda _=False, k=key: self._pick(k))
            self.edits[key], self.swatches[key] = ed, sw
            h = QHBoxLayout()
            h.setContentsMargins(0, 0, 0, 0)
            h.setSpacing(6)
            h.addWidget(sw)
            h.addWidget(ed)
            h.addStretch(1)
            row = QWidget()
            row.setLayout(h)
            form.addRow(title, row)
        root.addLayout(form)

        self._refresh_swatches()
        self._mark_scheme()                                   # 打开时按当前颜色预选方案

        tip = QLabel("颜色改完点「确定」立即生效（不需要重开工程）；"
                     "颜色是**软件级**设置（写软件根目录 app_settings.json，换工程不丢）。\n"
                     "「存为预设…」把当前全部颜色存成一个命名的方案（可跨次打开复用、可删除）；"
                     "内置方案（含「经典（默认）」＝程序默认）不可删除。\n"
                     "只存色相，**通透度按角色固定**：波形面板不透明、钢琴卷帘波形半透明、"
                     "PIT 线 alpha≈205，所以换色不会把波形/PIT 改糊。\n"
                     "口型面板背景**不参与自定义**（沿用软件主题色），只有曲线描边与填充两色可改。\n"
                     "休止符(灰)、连音音符(绿)、选中框(黄)是**语义标记色**，保持固定不参与自定义。")
        tip.setWordWrap(True)
        root.addWidget(tip)
        root.addStretch(1)

    # ---- 内部 ----
    def _rebuild_schemes(self):
        """重建方案下拉：自定义 / 内置方案（"b:"） / 自存预设（"c:"）。"""
        cur = self.cb_scheme.currentData() if self.cb_scheme.count() else None
        self.cb_scheme.blockSignals(True)
        self.cb_scheme.clear()
        self.cb_scheme.addItem("自定义", "")
        for name in COLOR_SCHEMES:
            self.cb_scheme.addItem(name, "b:" + name)
        for name in self.presets:
            self.cb_scheme.addItem("★ " + name, "c:" + name)
        if cur:
            i = self.cb_scheme.findData(cur)
            if i >= 0:
                self.cb_scheme.setCurrentIndex(i)
        self.cb_scheme.blockSignals(False)
        self.btn_del.setEnabled(bool(self.presets))

    def _scheme_colors(self, name: str, custom: bool) -> dict[str, str] | None:
        return (self.presets if custom else COLOR_SCHEMES).get(name)

    def _save_preset(self):
        name, ok = QInputDialog.getText(self, "存为配色预设", "预设名称：")
        if not ok or not name.strip():
            return
        name = name.strip()
        if name in COLOR_SCHEMES:
            QMessageBox.warning(self, "名称被占用", f"「{name}」是内置方案名，请换一个。")
            return
        self.presets[name] = dict(self.values())
        self._rebuild_schemes()
        i = self.cb_scheme.findData("c:" + name)
        if i >= 0:
            self.cb_scheme.setCurrentIndex(i)

    def _del_preset(self):
        data = self.cb_scheme.currentData() or ""
        if not data.startswith("c:"):
            QMessageBox.information(self, "无法删除", "请先在下拉里选中一个自存预设（★ 开头）。")
            return
        name = data[2:]
        if QMessageBox.question(self, "删除预设", f"删除配色预设「{name}」？") \
                != QMessageBox.StandardButton.Yes:
            return
        self.presets.pop(name, None)
        self._rebuild_schemes()
        self._mark_scheme()

    def _set_color(self, key: str, value: str):
        self.edits[key].setText(str(value))
        self._refresh_swatch(key)

    def _refresh_swatch(self, key: str):
        c = QColor(self.edits[key].text().strip())
        if not c.isValid():
            c = QColor(DEFAULT_COLORS[key])
        self.swatches[key].setStyleSheet(
            f"QPushButton {{ background: {c.name()}; border: 1px solid #454a57;"
            f" border-radius: 3px; }}")

    def _refresh_swatches(self):
        for key, _title in self.ROWS:
            self._refresh_swatch(key)

    def _pick(self, key: str):
        cur = QColor(self.edits[key].text().strip())
        if not cur.isValid():
            cur = QColor(DEFAULT_COLORS[key])
        c = QColorDialog.getColor(cur, self, "选择颜色")
        if c.isValid():
            self._set_color(key, c.name())
            self._mark_scheme()

    def _on_scheme(self, _idx: int):
        data = self.cb_scheme.currentData() or ""
        if not data:
            return                                            # 「自定义」：不动颜色
        cols = self._scheme_colors(data[2:], data.startswith("c:"))
        if not cols:
            return
        for k in COLOR_KEYS:
            self._set_color(k, cols.get(k, DEFAULT_COLORS[k]))

    def _mark_scheme(self):
        """四个颜色正好等于某方案（内置或自存）→ 下拉预选它；否则回到「自定义」。"""
        cur = self.values()
        hit: str | None = None
        _builtin = scheme_of(cur)
        if _builtin:
            hit = "b:" + _builtin
        else:
            for name, cols in self.presets.items():
                if all(cur[k].lower() == v.lower() for k, v in cols.items()):
                    hit = "c:" + name
                    break
        idx = self.cb_scheme.findData(hit) if hit else 0
        self.cb_scheme.blockSignals(True)
        self.cb_scheme.setCurrentIndex(max(0, idx))
        self.cb_scheme.blockSignals(False)

    def values(self) -> dict[str, str]:
        out: dict[str, str] = {}
        for key, _title in self.ROWS:
            c = QColor(self.edits[key].text().strip())
            out[key] = c.name() if c.isValid() else DEFAULT_COLORS[key]
        return out

    def apply_to_settings(self, settings: ProjectSettings) -> None:
        for key, hexv in self.values().items():
            setattr(settings, key, hexv)
        settings.color_presets = {k: dict(v) for k, v in self.presets.items()}


class SettingsDialog(QDialog):
    """统一设置：左栏导航 + 右栏详情。"""

    NAV = (("语言", "主语言（跟随工程）"), ("频谱", "频谱显示"),
           ("波形", "钢琴窗波形与 PIT 音高线"),
           ("面板", "面板上下顺序 / 显隐 / 口型纵轴"),
           ("配色", "音符 / PIT / 波形 / 口型颜色与配色方案"))

    def __init__(self, settings: ProjectSettings, project: Project | None = None, parent=None):
        super().__init__(parent)
        self.setWindowTitle("设置")
        self.setMinimumSize(900, 560)
        self.project = project
        self.app_settings = settings

        root = QVBoxLayout(self)

        body = QHBoxLayout()

        self.nav = QListWidget()
        self.nav.setFixedWidth(150)
        for key, _desc in self.NAV:
            self.nav.addItem(key)
        body.addWidget(self.nav, 0)

        self.stack = QStackedWidget()
        self.lang_page = LangPage(project, self)
        self.spec_page = SpecPage(settings, self)
        self.wave_page = WavePage(settings, self)
        self.panel_page = PanelPage(settings, self)
        self.palette_page = PalettePage(settings, self)
        self.pages: dict[str, QWidget] = {"语言": self.lang_page, "频谱": self.spec_page,
                                          "波形": self.wave_page, "面板": self.panel_page,
                                          "配色": self.palette_page}
        for key, _desc in self.NAV:
            self.stack.addWidget(self.pages[key])
        body.addWidget(self.stack, 1)
        self.nav.currentRowChanged.connect(self.stack.setCurrentIndex)
        self.nav.setCurrentRow(0)
        root.addLayout(body, 1)

        self.hint = QLabel("")
        self.hint.setWordWrap(True)
        root.addWidget(self.hint)
        self.nav.currentRowChanged.connect(self._sync_hint)
        self._sync_hint(0)

        bb = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        bb.button(QDialogButtonBox.Ok).setText("确定")
        bb.button(QDialogButtonBox.Cancel).setText("取消")
        bb.accepted.connect(self.accept)
        bb.rejected.connect(self.reject)
        root.addWidget(bb)

    def page(self, key: str) -> QWidget:
        return self.pages[key]

    def _sync_hint(self, row: int):
        if 0 <= row < len(self.NAV):
            self.hint.setText(f"<b>{self.NAV[row][0]}</b> — {self.NAV[row][1]}")

    def apply_to_settings(self):
        """把各页写进对象：语言（主语言/无声音素）→ **工程**；其余 → **软件级 ProjectSettings**。

        落盘由调用方负责（语言随 `project.save()`；界面设置随 `save_app_settings()`），
        这样不会因为写设置而覆盖布局记忆。
        """
        if self.project is not None:
            self.lang_page.apply_to_project(self.project)
        for page in (self.spec_page, self.wave_page, self.panel_page, self.palette_page):
            page.apply_to_settings(self.app_settings)
