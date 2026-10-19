# -*- coding: utf-8 -*-
"""
主窗口：
  菜单栏（工程：新建/打开/最近/保存/导出/回写 ＋ 编辑 ＋ 工具 ＋ 视图 ＋ 设置 ＋ 帮助）
  （工具栏已整条去掉：撤销重做在编辑菜单、推理与批量替换在工具菜单、配色在视图→频谱配色）
  + 顶部层信息栏：显示当前句、时长、选中音素/音符的详细信息
  + 左侧句子列表 │ 右侧多面板 QSplitter（波形 / 钢琴卷帘 / 口型 / mel 频谱，均可拖拽调整）
"""
from __future__ import annotations

import datetime
import pathlib
import re

import soundfile as sf
from PySide6.QtCore import Qt, QTimer, QSettings
from PySide6.QtGui import QAction, QActionGroup, QColor, QKeySequence, QShortcut
from PySide6.QtWidgets import (
    QMainWindow, QWidget, QListWidget, QListWidgetItem, QVBoxLayout, QHBoxLayout,
    QFileDialog, QMessageBox, QInputDialog, QComboBox, QSplitter,
)

from core.csv_io import write_csv
from core.curve import FPS as CURVE_FPS, MouthCurveExtractor, load_curve, save_curve
from core.edit import EditModel, note_labels, sync_word_marks, word_phone_counts, word_phone_groups
from core.export import ExportValidationError, export_dataset, unknown_report
from core.paths import MOUTH_MODEL, lang_dict
from core.phonemes import LANGS
from core.project import (
    DEFAULT_COLORS, KIND_REFURBISH, PANE_KEYS, PROJECT_FILE, REFURBISH_SUFFIX, Project,
    detect_kind, find_refurbish_project, load_app_settings, save_app_settings,
)
from core.sentence import SentenceData
from core.spectrogram import CMAP_LABELS, CMAPS, DEFAULT_CMAP, SpecData, spec_from_audio
from core.textgrid_io import parse_textgrid, tier_items
from engine.f0 import load_f0
from ui.audio import AudioEngine
from ui.settings_dialog import SettingsDialog
from ui.views import CurveCanvas, SpectroRollView, TierBar, ViewState, WaveformView
from ui.worker import InferenceWorker, MouthCurveWorker

BUILD = "2026.10.19-r50"    # 版本指纹（标题栏可见；改动未生效时先核对这个）

RECENT_MAX = 12

_EXPORT_AUDIO_DIR = "wavs"   # 数据集里音频子目录名（与既有数据集/上游 build_dataset 一致）

# 面板显隐：面板 key → 软件级设置字段 / 界面标题（「显示」菜单与 设置→「面板」页共用）
_PANEL_ATTR = {"wave": "wave_panel", "roll": "roll_panel",
               "curve": "curve_panel", "spec": "spec_panel"}
_PANEL_TITLE = {"wave": "波形面板", "roll": "钢琴卷帘",
                "curve": "口型面板", "spec": "频谱窗"}


def _panel_visibility(st) -> dict[str, bool]:
    """面板显隐（**软件级**设置：wave_panel / roll_panel / curve_panel / spec_panel）；
    无设置时全开。"""
    if st is None:
        return {k: True for k in _PANEL_ATTR}
    return {k: bool(getattr(st, attr, True)) for k, attr in _PANEL_ATTR.items()}


def _safe_name(name: str) -> str:
    """把用户输入的名字清成可用作文件夹名（去掉路径分隔符与 Windows 非法字符）。"""
    return re.sub(r'[\\/:*?"<>|]', "_", (name or "").strip())


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("DS-Labeler")
        self.resize(1560, 940)

        self.project: Project | None = None
        self.audio = AudioEngine()
        self.state = ViewState(duration=10.0)
        self.editor: EditModel | None = None
        self.spec: SpecData | None = None
        self._samples = None                     # 当前句采样（改频谱设置时重算用）
        self._sr: int = 0
        self._mouth: MouthCurveExtractor | None = None   # R3MOE 口型提取器（懒加载）
        self._curve: np.ndarray | None = None            # 当前句口型曲线（缓存/现算）
        self.mouth_worker: MouthCurveWorker | None = None  # 批量计算口型的后台线程
        self.tg: SentenceData | None = None      # 源 TextGrid（texts/words/phones 三层参照）
        self._sentences: list[str] = []
        self._current: str = ""
        # 切换/重建句子列表期间屏蔽选中事件：Qt 在 list.clear()/addItem() 时会自动
        # 发出 itemSelectionChanged（带旧行号、旧选中项），若照它去 _load_sentence，
        # 就会拿**上一份工程的句名**去新工程里找 wav（报"文件不存在/加载失败"），
        # 还会把新工程侧车的 last_sentence 污染成旧句名。
        self._switching: bool = False
        self.worker: InferenceWorker | None = None
        self._classifier = None                  # PhonemeClassifier（语言设置/标签用）
        self._settings = QSettings("DS-Labeler", "DS-Labeler")
        # 软件级设置 / 布局记忆（存软件根目录 app_settings.json，跨工程共用）
        self.app_settings, self.app_layout = load_app_settings()
        self._autosave_timer = QTimer(self)
        self._autosave_timer.setSingleShot(True)
        self._autosave_timer.setInterval(400)
        self._autosave_timer.timeout.connect(self._save_current)

        self._build_ui()
        self._build_menu()

        self._timer = QTimer(self)
        self._timer.setInterval(66)
        self._timer.timeout.connect(self._refresh_play_pos)

        # 软件级设置**无工程也生效**：启动即按 app_settings.json 挂面板 / 上色 / 波形参数
        self._sync_settings_from_project()
        self._apply_panes()
        self._apply_wave_metrics()
        self._apply_colors()
        self._apply_curve_options()
        self._apply_kind()             # 无工程时：层信息条按标准 3 行，导出/回写项置灰
        self._restore_layout()

        # 空格 = 播放/停止（窗口级快捷键：不依赖面板焦点；
        # 只在视图 keyPressEvent 里处理时，焦点在句子列表按空格无反应）
        QShortcut(QKeySequence(Qt.Key_Space), self, activated=self._toggle_play)

        # 版本指纹：标题栏可见，用于确认实际运行的构建（排查"改了但没生效"）
        import ui.views as _views_mod
        self.setWindowTitle(f"DS-Labeler · build {BUILD}")
        print(f"[boot] build={BUILD}\n[boot] views={_views_mod.__file__}", flush=True)

        self.statusBar().showMessage(
            f"build {BUILD} ｜ 打开工程开始工作（Ctrl+O，或工程菜单 → 新建工程）")

    # ================= UI 构建 =================
    def _build_ui(self):
        central = QWidget()
        root = QHBoxLayout(central)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        self.main_split = main_split = QSplitter(Qt.Horizontal)   # 左列表 │ 右面板（尺寸进布局记忆）

        # 左：句子列表（顶部排序：按名称 / 按标注质量分）
        left = QWidget()
        lv = QVBoxLayout(left)
        lv.setContentsMargins(0, 0, 0, 0)
        lv.setSpacing(0)
        self.sort_combo = QComboBox()
        self.sort_combo.addItems(["按名称", "按分数（低→高）", "按分数（高→低）"])
        self.sort_combo.currentIndexChanged.connect(lambda _: self._reload_sentence_list())
        lv.addWidget(self.sort_combo)
        self.list = QListWidget()
        self.list.itemSelectionChanged.connect(self._on_select_sentence)
        lv.addWidget(self.list, 1)
        main_split.addWidget(left)

        # 右：层信息栏 + 三面板
        right = QWidget()
        rv = QVBoxLayout(right)
        rv.setContentsMargins(0, 0, 0, 0)
        rv.setSpacing(0)

        self.tier_bar = TierBar(self.state)
        self.tier_bar.sig_phone_selected.connect(self._on_tier_phone_selected)
        self.tier_bar.sig_tier_selected.connect(self._on_tier_item_selected)
        self.tier_bar.sig_viewport.connect(self._sync_viewport)
        self.tier_bar.sig_changed.connect(self._on_edited)      # words/texts 编辑完成
        rv.addWidget(self.tier_bar)

        # 顺序：层信息栏（上）→ 波形（中）→ 钢琴卷帘 → 口型 → 频谱窗（下，可拉小）
        # 各面板共用同一 ViewState（同一浮点时间映射），所以时间轴逐像素对齐；
        # 挂载顺序与显隐由设置决定（panel_order + *_panel），由 _apply_panes 负责。
        self.panes = QSplitter(Qt.Vertical)
        self.view = SpectroRollView(self.state, layers=("notes",))       # 钢琴卷帘
        self.spec_view = SpectroRollView(self.state, layers=("spec",))   # mel 频谱窗
        self.wave = WaveformView(self.state)
        self.curve = CurveCanvas(self.state)                             # 口型（R3MOE 开口度）
        for v in (self.view, self.spec_view, self.wave, self.curve):
            v.sig_play_toggle.connect(self._toggle_play)
            v.sig_play_phoneme.connect(self._play_phoneme)
            v.sig_changed.connect(self._on_edited)
            v.sig_status.connect(lambda s: self.statusBar().showMessage(s))
            v.sig_selection.connect(self._on_view_selection)
            v.sig_viewport.connect(self._sync_viewport)
            v.sig_cursor.connect(self._on_cursor_moved)   # 光标竖直虚线跨面板同步
        for k, w in (("wave", self.wave), ("roll", self.view),
                     ("curve", self.curve), ("spec", self.spec_view)):
            if self.app_settings and bool(getattr(self.app_settings,
                                                  _PANEL_ATTR[k], True)):
                self.panes.addWidget(w)
        self.panes.setSizes([160, 360, 160, 240])
        self.panes.setChildrenCollapsible(False)
        rv.addWidget(self.panes, 1)
        main_split.addWidget(right)

        main_split.setSizes([240, 1320])
        main_split.setChildrenCollapsible(False)
        main_split.setStretchFactor(0, 0)
        main_split.setStretchFactor(1, 1)
        root.addWidget(main_split)
        self.setCentralWidget(central)

    def _build_menu(self):
        mb = self.menuBar()

        m_proj = mb.addMenu("工程(&P)")
        a_new = QAction("新建工程…", self)
        a_new.setShortcut("Ctrl+N")
        a_new.triggered.connect(self._new_project)
        m_proj.addAction(a_new)

        a_open = QAction("打开工程…", self)
        a_open.setShortcut("Ctrl+O")
        a_open.triggered.connect(self._open_project)
        m_proj.addAction(a_open)

        self.m_recent = m_proj.addMenu("最近工程")
        self._rebuild_recent_menu()

        m_proj.addSeparator()
        a_save = QAction("保存工程", self)
        a_save.setShortcut("Ctrl+S")
        a_save.triggered.connect(self._save_project)
        m_proj.addAction(a_save)
        m_proj.addSeparator()
        # 导出按工程类型分开：标准 → 数据集（raw/wavs + transcriptions.csv）；
        # 翻新 → 只输出一个 csv。两类工程共用一套菜单，按 kind 启用其一。
        self.a_export = QAction("导出数据集…", self)
        self.a_export.triggered.connect(self._export)
        m_proj.addAction(self.a_export)
        self.a_export_csv = QAction("导出 CSV…", self)
        self.a_export_csv.triggered.connect(self._export_csv)
        m_proj.addAction(self.a_export_csv)
        self.a_tg = QAction("回写 TextGrid…", self)
        self.a_tg.triggered.connect(self._writeback_tg)
        m_proj.addAction(self.a_tg)
        m_proj.addSeparator()
        a_quit = QAction("退出", self)
        a_quit.setShortcut("Ctrl+Q")
        a_quit.triggered.connect(self.close)
        m_proj.addAction(a_quit)

        m_edit = mb.addMenu("编辑(&E)")
        self.act_undo_menu = QAction("撤销", self)
        self.act_undo_menu.setShortcut("Ctrl+Z")
        self.act_undo_menu.triggered.connect(self._undo)
        m_edit.addAction(self.act_undo_menu)
        self.act_redo_menu = QAction("重做", self)
        self.act_redo_menu.setShortcuts(["Ctrl+Y", "Ctrl+Shift+Z"])
        self.act_redo_menu.triggered.connect(self._redo)
        m_edit.addAction(self.act_redo_menu)

        # 工具菜单：原先挂在工具栏上的三项都收进这里（工具栏整条已去掉）
        m_tool = mb.addMenu("工具(&T)")
        a_batch = QAction("批量预处理", self)
        a_batch.triggered.connect(self._batch_infer)
        m_tool.addAction(a_batch)
        a_retry = QAction("重跑当前句", self)
        a_retry.triggered.connect(self._rerun_current)
        m_tool.addAction(a_retry)
        a_replace = QAction("批量替换音素…", self)
        a_replace.triggered.connect(self._batch_replace)
        m_tool.addAction(a_replace)
        m_tool.addSeparator()
        # 口型（R3MOE）：后台批量算出全部句的口型并缓存（切句时按需计算之外的批量入口）
        self.act_batch_mouth = QAction("计算所有文件口型", self)
        self.act_batch_mouth.triggered.connect(self._batch_mouth)
        m_tool.addAction(self.act_batch_mouth)

        m_view = mb.addMenu("视图(&V)")
        a_fit = QAction("适配全曲", self)
        a_fit.setShortcut("Z")
        a_fit.triggered.connect(self._fit_view)
        m_view.addAction(a_fit)
        a_zin = QAction("放大", self)
        a_zin.triggered.connect(lambda: self._zoom(1.3))
        m_view.addAction(a_zin)
        a_zout = QAction("缩小", self)
        a_zout.triggered.connect(lambda: self._zoom(1 / 1.3))
        m_view.addAction(a_zout)
        m_view.addSeparator()
        a_eq = QAction("面板均分", self)
        a_eq.triggered.connect(self._reset_panes)
        m_view.addAction(a_eq)
        m_view.addSeparator()
        self.act_wave = QAction("波形预览", self)
        self.act_wave.setCheckable(True)
        self.act_wave.triggered.connect(self._on_wave_toggle)
        m_view.addAction(self.act_wave)
        m_cmap = m_view.addMenu("频谱配色")
        self._cmap_group = QActionGroup(self)
        self._cmap_group.setExclusive(True)
        for name in CMAPS:
            act = QAction(CMAP_LABELS.get(name, name), self)
            act.setCheckable(True)
            act.setData(name)
            act.triggered.connect(self._on_cmap_action)
            self._cmap_group.addAction(act)
            m_cmap.addAction(act)
        self._cmap_group.actions()[0].setChecked(True)

        # 「显示」菜单：各面板的显隐 + 口型纵轴（软件级设置，与 设置→「面板」页同一份）
        m_disp = mb.addMenu("显示(&D)")
        self.act_wave_panel = QAction("波形面板", self)
        self.act_wave_panel.setCheckable(True)
        self.act_wave_panel.triggered.connect(self._on_wave_panel_toggle)
        m_disp.addAction(self.act_wave_panel)
        self.act_roll_panel = QAction("钢琴卷帘", self)
        self.act_roll_panel.setCheckable(True)
        self.act_roll_panel.triggered.connect(
            lambda c: self._on_panel_visibility("roll", c))
        m_disp.addAction(self.act_roll_panel)
        self.act_curve_panel = QAction("口型面板", self)
        self.act_curve_panel.setCheckable(True)
        self.act_curve_panel.triggered.connect(
            lambda c: self._on_panel_visibility("curve", c))
        m_disp.addAction(self.act_curve_panel)
        self.act_spec_panel = QAction("频谱窗", self)
        self.act_spec_panel.setCheckable(True)
        self.act_spec_panel.triggered.connect(
            lambda c: self._on_panel_visibility("spec", c))
        m_disp.addAction(self.act_spec_panel)
        m_disp.addSeparator()
        # 口型面板的纵轴标尺（显示菜单里给个快捷开关，等价于 设置→「面板」页里那一项）
        self.act_curve_fit = QAction("口型按峰值自适应高度", self)
        self.act_curve_fit.setCheckable(True)
        self.act_curve_fit.triggered.connect(self._on_curve_autoscale)
        m_disp.addAction(self.act_curve_fit)

        m_set = mb.addMenu("设置(&S)")
        a_set = QAction("设置…", self)
        a_set.triggered.connect(self._open_settings)
        m_set.addAction(a_set)

        m_help = mb.addMenu("帮助(&H)")
        a_about = QAction("关于 / 快捷键", self)
        a_about.triggered.connect(self._about)
        m_help.addAction(a_about)

    # ================= 最近工程 =================
    def _load_recent(self) -> list[str]:
        val = self._settings.value("recent/projects", [])
        if isinstance(val, str):
            val = [val]
        return [str(v) for v in (val or [])]

    def _push_recent(self, path: pathlib.Path):
        path = pathlib.Path(path)
        items = [p for p in self._load_recent() if pathlib.Path(p) != path]
        items.insert(0, str(path))
        self._settings.setValue("recent/projects", items[:RECENT_MAX])
        self._rebuild_recent_menu()

    def _rebuild_recent_menu(self):
        if not hasattr(self, "m_recent"):
            return
        self.m_recent.clear()
        items = self._load_recent()
        if not items:
            act = QAction("（无）", self)
            act.setEnabled(False)
            self.m_recent.addAction(act)
            return
        for p in items:
            q = pathlib.Path(p)
            # 翻新工程的近项记的是 `<名>.refurbish.json` 文件 → 前缀标注类型，避免混淆
            tag = "翻新 · " if str(p).endswith(REFURBISH_SUFFIX) else ""
            act = QAction(f"{tag}{q.name}  —  {p}", self)
            act.setEnabled(q.exists())
            act.triggered.connect(lambda _=False, qq=p: self._open_project_at(qq))
            self.m_recent.addAction(act)
        self.m_recent.addSeparator()
        clear = QAction("清空列表", self)
        clear.triggered.connect(lambda: (self._settings.setValue("recent/projects", []),
                                         self._rebuild_recent_menu()))
        self.m_recent.addAction(clear)

    # ================= 工程 =================
    def _new_project(self):
        """新建工程：选文件夹 → 按**目录结构**判类型 → **手动**选主语言（不做任何猜测）。"""
        folder = QFileDialog.getExistingDirectory(
            self, "选择新工程文件夹（标准：wav/ + TextGrid/；翻新：transcriptions.csv + wavs/）")
        if not folder:
            return
        root = pathlib.Path(folder)
        if (root / PROJECT_FILE).exists():
            QMessageBox.information(self, "已有工程", "该文件夹已是标准工程，将直接打开。")
            self._open_project_at(folder)
            return
        existing = find_refurbish_project(root)
        if existing is not None:
            QMessageBox.information(self, "已有工程", "该文件夹已建过翻新工程，将直接打开。")
            self._activate_project(existing)
            return
        kind = detect_kind(root)
        if kind is None:
            QMessageBox.warning(
                self, "无法新建",
                "无法识别工程类型：\n"
                "· 标准工程需 wav/ 与 TextGrid/ 子文件夹；\n"
                "· 翻新工程需 transcriptions.csv 与 wavs/（或 wav/）。")
            return
        proj = self._create_project_ask_lang(root, kind)
        if proj is not None:
            self._activate_project(proj)

    def _create_project_ask_lang(self, root: pathlib.Path, kind: str) -> Project | None:
        """手动选主语言后建工程（标准 / 翻新都问；绝不自动猜）。"""
        lang, ok = QInputDialog.getItem(self, "新建工程", "主语言：", list(LANGS), 0, False)
        if not ok:
            return None
        try:
            proj = Project.create(root, name=root.name, main_lang=lang, kind=kind)
        except Exception as e:
            QMessageBox.critical(self, "新建失败", str(e))
            return None
        label = "翻新" if kind == KIND_REFURBISH else "标准"
        self._log_static(proj, f"新建{label}工程（主语言 {lang}）")
        return proj

    def _open_project(self):
        """打开**工程文件**：打开工程选文件，新建工程才选文件夹。"""
        path, _ = QFileDialog.getOpenFileName(
            self, "打开 DS-Labeler 工程（选择 project.json 或 *.refurbish.json）",
            self._project_dir_hint(),
            f"DS-Labeler 工程 ({PROJECT_FILE} *{REFURBISH_SUFFIX});;"
            f"标准工程 ({PROJECT_FILE});;翻新工程 (*{REFURBISH_SUFFIX});;"
            f"JSON 文件 (*.json);;所有文件 (*)")
        if not path:
            return
        self._open_project_at(path)

    def _project_dir_hint(self) -> str:
        """文件对话框的起始目录：当前工程根目录 / 最近工程所在目录。"""
        if self.project is not None:
            return str(self.project.root)
        for p in self._load_recent():
            q = pathlib.Path(p)
            if q.is_dir():
                return str(q)
            if q.is_file():
                return str(q.parent)
        return ""

    def _open_project_at(self, path: str):
        """打开工程。path 可以是**工程文件夹**、`project.json`，或 `<名>.refurbish.json`。

        （文件夹选择器在 Windows 上把用户选中的文件也当成路径返回过，
          所以这两种形态都接受，避免"选了工程文件却打不开、还停在上一份工程"。）
        """
        p = pathlib.Path(path)
        try:
            if not p.exists():
                QMessageBox.warning(self, "无法打开", f"路径不存在：{p}")
                self._rebuild_recent_menu()
                return
            if p.is_file() and p.name.endswith(REFURBISH_SUFFIX):
                proj = Project.load(p)                       # 翻新工程文件
            else:
                # 选中的是文件 → 工程根就是它所在目录（只认 project.json；
                # 选 wav/txt 等非工程文件会明确报错）
                root = p.parent if p.is_file() else p
                if (root / PROJECT_FILE).exists():
                    proj = Project.load(root)
                elif p.is_file():
                    QMessageBox.warning(
                        self, "无法打开",
                        f"{p.name} 不是工程文件"
                        f"（请选择 {PROJECT_FILE} 或 *{REFURBISH_SUFFIX}）。")
                    self._rebuild_recent_menu()
                    return
                else:
                    proj = self._adopt_folder(root)
                    if proj is None:
                        self._rebuild_recent_menu()
                        return
        except Exception as e:
            QMessageBox.critical(self, "打开失败", str(e))
            self._rebuild_recent_menu()
            return
        if not self._ensure_main_lang(proj):     # 主语言必填（老工程可能没写）
            self._rebuild_recent_menu()
            return
        self._activate_project(proj)

    def _adopt_folder(self, root: pathlib.Path) -> Project | None:
        """把一个还不是工程的文件夹「认领」为工程：按目录结构判类型 + 手动选主语言。

        已建过翻新工程（软件根 `refurbish/` 里有指向它的工程）→ 直接复用，不重复建。
        """
        existing = find_refurbish_project(root)
        if existing is not None:
            return existing
        kind = detect_kind(root)
        if kind is None:
            QMessageBox.warning(
                self, "无法打开",
                f"文件夹内没有 {PROJECT_FILE}，也不是可识别的新数据：\n"
                "· 标准工程需 wav/ 与 TextGrid/ 子文件夹；\n"
                "· 翻新工程需 transcriptions.csv 与 wavs/（或 wav/）。")
            return None
        return self._create_project_ask_lang(root, kind)

    def _ensure_main_lang(self, proj: Project) -> bool:
        """主语言**必填**：打开工程时若尚未设定，先强制补选，否则不打开。

        新建工程（含翻新工程）都必须先设定主语言才能打开：不做任何猜测，
        `Project.load` 也不再静默回退 zh（缺字段读成空串，到这里必须补上）。补选成功后
        写回工程（标准 → 侧车；翻新 → 软件根工程文件），下次不再问。
        """
        if proj.main_lang in LANGS:
            return True
        lang, ok = QInputDialog.getItem(
            self, "设定主语言",
            f"工程「{proj.name}」尚未设定主语言。\n"
            "音素元辅分类与 ph_num 都按主语言计算，必须先指定：",
            list(LANGS), 0, False)
        if not ok or lang not in LANGS:
            QMessageBox.warning(self, "需要主语言", "未设定主语言，已取消打开该工程。")
            return False
        proj.main_lang = lang
        try:
            proj.save()
        except Exception:
            pass
        self._log_static(proj, f"设定主语言：{lang}")
        return True

    def _activate_project(self, proj: Project):
        self._save_current()          # 切工程前先存
        if self.project is not None:  # 记下旧工程的布局（面板长宽 / 钢琴窗纵高）
            self._capture_layout()
            try:
                self.project.save()
            except Exception:
                pass
        self.project = proj
        # ✓ 标记只属于「人工编辑过」的句子（编辑必有自动保存的工作文件）。
        # 旧版曾把「预处理完成」也写进 edited，打开时把没有工作文件的条目剔掉。
        _gone = {n for n in proj.edited if not proj.sentence_work_file(n).exists()}
        if _gone:
            proj.edited -= _gone
            self._log(f"剔除 {len(_gone)} 句误标的「已编辑」标记（实为预处理完成，非人工编辑）")
            try:
                proj.save()
            except Exception:
                pass
        # 界面设置与布局是**软件级**的：把工程接到同一份对象上，
        # 于是各处的 project.settings / project.save() 都自然写向 app_settings.json
        proj.settings = self.app_settings
        proj.layout = self.app_layout
        for v in (self.view, self.spec_view, self.wave, self.curve):
            v.set_model(None)
        for v in (self.view, self.spec_view):
            v.set_spec(None)
        self.view.set_f0(None)
        self.wave.set_audio(None, 44100)
        self.view.set_audio(None, 44100)
        self._set_curve(None)
        self.editor = None
        self.spec = None
        self.tg = None
        self._current = ""
        # 最近工程记**工程文件**：标准 = <根>/project.json；翻新 = refurbish/<名>.refurbish.json
        # （翻新的数据目录是上游领地，不能把"已建工程"的状态写进那里）
        self._push_recent(proj.project_file)
        self._sync_settings_from_project()
        self._apply_panes()            # 按 panel_order 挂载面板 + 波形面板显隐
        self._apply_wave_metrics()     # 钢琴窗波形上下宽度 / 段落外扩时长 / PIT 线粗细
        self._apply_colors()           # 音符 / PIT / 两处波形的颜色
        self._apply_curve_options()    # 口型纵轴自适应
        self._apply_kind()             # 层信息条行数 + 导出/回写项启用态（按工程类型）
        self._restore_layout()         # 恢复上次关闭时的面板长宽 / 钢琴窗纵高
        # 音符标签器（惯例：前面的辅音 + 当前的元音）
        self._classifier = proj.make_classifier()
        self.view.labeler = lambda md: note_labels(md, self._classifier)
        self._diagnosis = proj.load_diagnosis()     # 标注质量指标（TextGrid/statistics）
        self._reload_sentence_list()

    def _apply_kind(self):
        """按工程类型调整 UI 细节：层信息条显示哪几行 + 导出/回写菜单项启用态。

        两类工程**共用同一套 UI**，唯一区别是翻新工程没有 texts/words 两层参照数据
        （层信息条只留 phones），且导出走「导出 CSV」、不支持回写 TextGrid。
        """
        p = self.project
        ref = p is not None and p.is_refurbish
        std = p is not None and not p.is_refurbish
        self.tier_bar.set_tier_names(("phones",) if ref else ("texts", "words", "phones"))
        self.a_export.setEnabled(std)        # 标准 → 导出数据集
        self.a_export_csv.setEnabled(ref)    # 翻新 → 导出 CSV
        self.a_tg.setEnabled(std)            # 回写 TextGrid 仅标准工程可用

    def _sync_settings_from_project(self):
        """按**软件级**设置刷新工具栏/视图（无工程也调用，启动即生效）。"""
        st = self.app_settings
        cmap = st.spectrum_preset if st.spectrum_preset in CMAPS else DEFAULT_CMAP
        for v in (self.view, self.spec_view):
            v.set_cmap(cmap)
        wave = bool(st.wave_show)
        self.view.set_wave(wave)
        self.act_wave.setChecked(wave)
        self._sync_panel_actions()
        self._apply_wave_metrics()          # 波形宽度 / 外扩时长 / PIT 线粗细
        self._apply_colors()               # 音符 / PIT / 两处波形的颜色
        self._apply_curve_options()        # 口型纵轴自适应
        for act in self._cmap_group.actions():   # 视图→频谱配色 的勾选态跟着当前配色走
            if act.data() == cmap:
                act.setChecked(True)

    def _open_settings(self):
        """统一设置面板（左栏导航：语言 / 频谱 / 波形 / 面板 / 配色）。

        **无工程也可打开**：主语言页会置灰（语言随工程走），其余页照常可用——
        因为频谱 / 波形 / 面板 / 配色是**软件级**设置（写软件根目录 app_settings.json）。
        确定后：重建分类器与音符标签、按新顺序重挂面板，并按频谱设置重算当前句。
        """
        dlg = SettingsDialog(self.app_settings, self.project, self)
        if not dlg.exec():
            return
        dlg.apply_to_settings()
        save_app_settings(self.app_settings, self.app_layout)
        if self.project is not None:
            self._refresh_language_state()
        self._apply_panes()                         # 面板上下顺序 / 波形面板显隐立即生效
        st = self.app_settings
        self._sync_panel_actions()
        self._apply_wave_metrics()                  # 波形宽度 / 外扩时长 / PIT 线粗细
        self._apply_colors()                        # 音符 / PIT / 两处波形颜色立即生效
        self._apply_curve_options()                 # 口型纵轴自适应立即生效
        self._set_cmap(st.spectrum_preset)          # 同步工具栏/视图菜单配色
        if self._samples is not None and self._sr:
            self.spec = self._compute_spec(self._samples, self._sr)
            for v in (self.view, self.spec_view):
                v.set_spec(self.spec)
        lang = self.project.main_lang if self.project is not None else "-"
        self._log(f"设置已保存（软件级）：主语言 {lang} ｜ 面板顺序"
                  f"{'>'.join(st.panel_order)} ｜ 频谱窗={st.spec_window} 轴={st.spec_scale} "
                  f"hop={st.spec_hop} max={st.spec_max_freq:g}Hz 分辨率={st.spec_mel_res:g} "
                  f"dB=[{st.spec_db_min:g},{st.spec_db_max:g}] 配色={st.spectrum_preset} "
                  f"｜ PIT={st.pit_width:g}px ｜ 颜色 音符={st.note_color} "
                  f"PIT={st.pit_color} 波形={st.wave_panel_color}/{st.wave_roll_color} "
                  f"｜ 自存配色预设 {len(st.color_presets)} 套")
        self.statusBar().showMessage(
            "设置已保存（语言随工程 / 其余跨工程共用），当前句频谱已重算", 5000)

    def _apply_wave_metrics(self):
        """把**软件级**设置里的「波形上下最大宽度 / 每段外扩时长 / PIT 线粗细 / 标签字号
        / 音符块内文字开关」推给钢琴卷帘。"""
        st = self.app_settings
        self.view.set_wave_metrics(st.wave_h, st.wave_ext_ms)
        self.view.set_pit_width(st.pit_width)
        self.view.set_note_label_inner(st.note_label_inner)
        for v in (self.view, self.spec_view):        # 两窗共用同一套标签字号（音素名/边界标注）
            v.set_label_font(st.label_font_size)

    def _apply_colors(self):
        """把**软件级**设置里的「音符 / PIT / 两处波形 / 口型两色」颜色推给各视图
        （设置→「配色」页）。

        三个波形视图都推：波形面板看 wave_panel_color、钢琴卷帘看 wave_roll_color，
        音符 / PIT 只有钢琴卷帘画（推给频谱窗无副作用，保持状态一致）；口型两色推给口型画布。
        """
        st = self.app_settings
        note = st.note_color if st is not None else DEFAULT_COLORS["note_color"]
        pit = st.pit_color if st is not None else DEFAULT_COLORS["pit_color"]
        wp = st.wave_panel_color if st is not None else DEFAULT_COLORS["wave_panel_color"]
        wr = st.wave_roll_color if st is not None else DEFAULT_COLORS["wave_roll_color"]
        cl = st.curve_line_color if st is not None else DEFAULT_COLORS["curve_line_color"]
        cf = st.curve_fill_color if st is not None else DEFAULT_COLORS["curve_fill_color"]
        self.wave.set_wave_color(wp)
        for v in (self.view, self.spec_view):
            v.set_wave_color(wr)
            v.set_note_colors(note)
            v.set_pit_color(pit)
        self.curve.set_curve_colors(cl, cf)

    def _refresh_language_state(self):
        """语言表/音素系统/全局音素/主语言变化后：重建分类器与音符标签。"""
        self._classifier = self.project.make_classifier()
        self.view.labeler = lambda md: note_labels(md, self._classifier)
        for v in (self.view, self.spec_view, self.wave):
            v.update()
        self.tier_bar.update()
        sys_txt = "／".join(f"{lg}:{self.project.system_of(lg)}" for lg in LANGS)
        self._log(f"语言设置已保存：主语言 {self.project.main_lang}（系统 {sys_txt}）")
        self.statusBar().showMessage(
            f"语言设置已保存：主语言 {self.project.main_lang}", 5000)

    def _reload_sentence_list(self):
        # 重建期间屏蔽选中事件（Qt 会在 clear()/addItem() 时自动发 itemSelectionChanged，
        # 带的是旧行号/旧选中项 → 会拿上一份工程的句名去新工程里找 wav 而报"加载失败"）
        self._switching = True
        try:
            self.list.clear()
            self._sentences = self.project.list_sentences()
            # 排序：按名称 / 按标注质量分（TextGrid/statistics/diagnosis.json，四项指标均值）
            mode = self.sort_combo.currentIndex() if hasattr(self, "sort_combo") else 0
            if mode in (1, 2):
                def key(n: str):
                    sc = Project.diagnosis_score(self._diagnosis.get(n))
                    if sc is None:
                        return (1, 0.0, n)                    # 无分句恒排最后
                    return (0, sc if mode == 1 else -sc, n)
                names = sorted(self._sentences, key=key)
            else:
                names = list(self._sentences)
            for name in names:
                sc = Project.diagnosis_score(self._diagnosis.get(name))
                item = QListWidgetItem(self._item_text(name))
                item.setData(Qt.UserRole, name)
                item.setToolTip(self._diagnosis_tooltip(name))
                if sc is not None and sc < 0.85:
                    item.setForeground(QColor("#ff9d6b"))    # 低分句标暖色，便于先审
                elif self._is_ready(name):
                    item.setForeground(Qt.gray)
                self.list.addItem(item)
        finally:
            self._switching = False
        self._bind_worker()
        missing = [n for n in self._sentences if not self._is_ready(n)]
        self.statusBar().showMessage(
            f"工程：{self.project.name}｜主语言 {self.project.main_lang}｜"
            f"{len(self._sentences)} 句｜待预处理 {len(missing)}")
        self._log(f"打开工程 {self.project.name}（{len(self._sentences)} 句，待预处理 {len(missing)}）· build {BUILD}")
        if self._sentences:
            # 记忆上次停留的句（工程侧车）；无记忆或已不在列表中 → 回落第一句
            # 行号必须按**列表显示顺序**取（按分数排序时显示序 ≠ 名称序）
            want = self.project.last_sentence
            row = names.index(want) if want in names else 0
            self.list.setCurrentRow(row)   # 触发 _load_sentence，当前句插队优先
            # 重建期间 Qt 可能已把当前行设成 row（事件被屏蔽）→ setCurrentRow 不再发信号，
            # 这时必须显式加载，否则打开工程后停在空白界面（"什么都没载入"）
            if self._current != names[row]:
                self._load_sentence(names[row])
        if missing and self.worker is not None:
            self.worker.enqueue(missing)

    def _diagnosis_tooltip(self, name: str) -> str:
        m = self._diagnosis.get(name)
        if not m:
            return "（无标注质量诊断：TextGrid/statistics/diagnosis.json 缺失该句）"
        sc = Project.diagnosis_score(m)
        return (f"标注质量 {sc:.4f}\n"
                f"confidence    {m.get('confidence', '-')}\n"
                f"determinacy   {m.get('determinacy', '-')}\n"
                f"monotonicity  {m.get('monotonicity', '-')}\n"
                f"agreement     {m.get('agreement', '-')}\n"
                f"skipped tokens {m.get('num_skipped_tokens', '-')}")

    def _item_text(self, name: str) -> str:
        """列表项文本：名称 + 质量分；**已人工编辑过的句前置 ✓ 标记**。"""
        sc = Project.diagnosis_score(self._diagnosis.get(name))
        base = f"{name}　{sc:.2f}" if sc is not None else name
        edited = self.project is not None and name in self.project.edited
        return f"✓ {base}" if edited else base

    def _list_item_named(self, name: str):
        """按 UserRole 存的句名找列表项（item 文本带分数后缀，不能按 text 匹配）。"""
        for i in range(self.list.count()):
            if self.list.item(i).data(Qt.UserRole) == name:
                return self.list.item(i)
        return None

    def _bind_worker(self):
        if self.worker is None:
            self.worker = InferenceWorker(self.project, self)
            self.worker.sig_progress.connect(self._on_infer_progress)
            self.worker.sig_sentence.connect(self._on_sentence_ready)
            self.worker.sig_failed.connect(self._on_infer_failed)
            self.worker.sig_provider.connect(self._on_provider)
            self.worker.sig_idle.connect(self._on_infer_idle)
        else:
            self.worker.set_project(self.project)
        # 打开工程即预建 ONNX 会话（+跑一次哑推理），把首次 3s 的建会话开销
        # 挪到用户看界面的空闲时间，第一句直接按稳态速度跑。
        self.worker.warmup()
        self._bind_mouth_worker()

    def _bind_mouth_worker(self):
        """口型线程（QThread 本体很轻，会话/缓存只在实际 enqueue 后才有开销）。

        真正的懒加载在 `MouthCurveWorker.run()`（权重缺失即退出）；且只有口型面板**可见**
        时主窗口才会 enqueue → 默认关面板不产生任何口型计算。
        """
        if self.mouth_worker is None:
            self.mouth_worker = MouthCurveWorker(self.project, self)
            self.mouth_worker.sig_progress.connect(self._on_mouth_progress)
            self.mouth_worker.sig_curve.connect(self._on_curve_ready)
            self.mouth_worker.sig_failed.connect(self._on_mouth_failed)
            self.mouth_worker.sig_idle.connect(self._on_mouth_idle)
        else:
            self.mouth_worker.set_project(self.project)

    def _is_ready(self, name: str) -> bool:
        f0 = (self.project.f0_dir / f"{name}.npz").exists()
        if self.project.is_refurbish:
            return f0                      # 翻新：音符来自 csv，只有 F0 需缓存
        return self.project.sentence_work_file(name).exists() and f0

    # ================= 句子加载 =================
    def _on_select_sentence(self):
        if self._switching:      # 列表重建中：这是 Qt 自动发出的选中事件，不当用户操作
            return
        items = self.list.selectedItems()
        if not items:
            return
        name = items[0].data(Qt.UserRole)
        if name == self._current:
            return
        self._save_current()
        if self.project is not None:
            try:
                self.project.save()          # 持久化 last_sentence / edited（切句即存）
            except Exception:
                pass
        self._load_sentence(name)

    def _load_sentence(self, name: str):
        try:
            wav_path = self.project.wav_file(name)
            samples, sr = sf.read(str(wav_path), dtype="float32", always_2d=True)
            samples = samples.mean(axis=1) if samples.shape[1] > 1 else samples[:, 0]

            self.audio.load(wav_path)
            self.state.set_duration(len(samples) / sr)
            self.wave.set_audio(samples, sr)
            self.view.set_audio(samples, sr)      # 钢琴卷帘的「音符波形」层用同一份采样
            # 播放头常驻：切句后仍停留在 0 点可见（三面板都要画）
            for v in (self.view, self.spec_view, self.wave):
                v.set_play_pos(0.0)
            self._current = name
            if self.project is not None:
                self.project.last_sentence = name

            # 频谱（整段一次算完并缓存；频率轴/参数取自工程设置 → 可自定义）
            self._samples, self._sr = samples, int(sr)
            self.spec = self._compute_spec(samples, sr)
            for v in (self.view, self.spec_view):
                v.set_spec(self.spec)
            st_spec = self.project.settings if self.project else None
            if self.spec is not None and st_spec is not None:
                self._log(f"[频谱] build={BUILD} 窗={self.spec.window} fft={self.spec.n_fft} "
                          f"hop={st_spec.spec_hop} dB=[{st_spec.spec_db_min:g},{st_spec.spec_db_max:g}] "
                          f"轴={self.spec.scale}/{st_spec.spec_mel_res:g} 行={self.spec.n_rows} "
                          f"max={int(self.spec.data.max())} mean={float(self.spec.data.mean()):.1f}")

            # 源数据：标准 → TextGrid 三层；翻新 → 无 TextGrid（只有 phones 层），源数据取自 csv
            self.tg = None if self.project.is_refurbish else self._read_textgrid(name)
            data = self._read_work(name)
            if data is None:
                if self.project.is_refurbish:
                    data = (self.project.source_sentence(name)
                            or SentenceData(name=name, phones=[], notes=[]))
                else:
                    data = SentenceData(name=name, phones=self._phones_of(self.tg), notes=[])
            self._seed_tiers(data)              # words/texts 层：源 TG 播种（标准工程）
            self.editor = EditModel(data)
            sync_word_marks(self.editor)        # word 名：音素集合变了才改写，否则保留源标记
            for v in (self.view, self.spec_view, self.wave, self.curve):
                v.set_model(self.editor)
            self.tier_bar.set_model(self.editor)
            self._load_f0(name)
            self._load_curve(name)
            self._update_label()
            self.audio.stop()
            self._timer.start()

            if not self._is_ready(name) and self.worker is not None:
                self.worker.enqueue([name], front=True, skip_cached=False)
        except Exception as e:
            self._log(f"加载失败 {name}: {e}")
            QMessageBox.warning(self, "加载失败", f"{name}: {e}")

    def _compute_spec(self, samples, sr) -> SpecData | None:
        """按工程设置计算整段频谱（窗长自适应；频率轴/分辨率/范围/强度均可自定义）。"""
        try:
            st = self.app_settings          # 频谱参数是软件级设置（无工程也有效）
            return spec_from_audio(
                samples, sr, hop=int(st.spec_hop), window=st.spec_window,
                max_freq=float(st.spec_max_freq), mel_res=float(st.spec_mel_res),
                scale=st.spec_scale,
                db_min=float(st.spec_db_min), db_max=float(st.spec_db_max))
        except Exception as e:
            self._log(f"频谱计算失败: {e}")
            return None

    # ================= 口型（R3MOE）=================
    def _curve_visible(self) -> bool:
        return bool(self.app_settings and self.app_settings.curve_panel)

    def _curve_path(self, name: str):
        return self.project.curve_dir / f"{name}.npz"

    def _set_curve(self, curve, fps: float = CURVE_FPS) -> None:
        self._curve = curve
        self.curve.set_curve(curve, fps)

    def _load_curve(self, name: str) -> None:
        """切句：缓存命中直接用；缺缓存且口型面板可见 → 交给后台现算并写缓存。

        面板默认关 → 不打扰（不开面板不算口型）。
        """
        self._set_curve(None)
        if self.project is None:
            return
        path = self._curve_path(name)
        if path.exists():
            try:
                c, fps = load_curve(path)
                self._set_curve(c, fps)
                return
            except Exception as e:
                self._log(f"口型缓存损坏 {name}: {e}（将重算）")
        if not self._curve_visible():
            return
        self._enqueue_mouth([name], front=True)

    def _enqueue_mouth(self, names, front: bool = False) -> int:
        if self.project is None:
            return 0
        if self.mouth_worker is None:
            self._bind_mouth_worker()
        if self.mouth_worker is None:      # 绑定被禁用（异常）→ 静默跳过
            return 0
        n = self.mouth_worker.enqueue(names, front=front)
        if n:
            self.statusBar().showMessage(f"正在计算口型（{n} 句）…")
        return n

    def _on_mouth_progress(self, name: str, done: int, total: int):
        self.statusBar().showMessage(f"计算口型 {name}（{done}/{total}）")

    def _on_curve_ready(self, name: str, curve, fps: float):
        if name == self._current:
            self._set_curve(curve, fps)

    def _on_mouth_failed(self, name: str, err: str):
        self._log(f"口型计算失败 {name}: {err}")
        self.statusBar().showMessage(f"口型计算失败：{err}", 6000)

    def _on_mouth_idle(self):
        self.statusBar().showMessage("口型计算完成", 5000)
        self._log("口型计算完成")

    def _batch_mouth(self):
        """菜单「计算所有文件口型」：后台批量算出全部句的口型并缓存（进度到状态栏）。"""
        if self.project is None or not self._sentences:
            self.statusBar().showMessage("没有打开工程，无法计算口型", 4000)
            return
        if not MOUTH_MODEL.exists():
            QMessageBox.information(
                self, "缺少口型模型",
                "未找到口型模型权重：\n"
                f"{MOUTH_MODEL}\n\n"
                "R3MOE 权重（opec.onnx）需自行从上游项目获取后放到该路径：\n"
                "https://github.com/KakaruHayate/R3MOE\n"
                "（本软件不打包该权重；详见 README「模型放置」一节）")
            return
        n = self._enqueue_mouth(self._sentences)
        self.statusBar().showMessage(
            f"已加入口型计算队列：{n} 句" if n else "所有句的口型缓存都已存在")

    def _read_work(self, name: str) -> SentenceData | None:
        path = self.project.sentence_work_file(name)
        if not path.exists():
            return None
        try:
            return SentenceData.load(path, name)
        except Exception as e:
            self._log(f"工作文件损坏 {name}: {e}（将重新生成）")
            return None

    def _read_textgrid(self, name: str) -> SentenceData | None:
        try:
            return parse_textgrid(self.project.tg_dir / f"{name}.TextGrid", name)
        except Exception as e:
            self._log(f"TextGrid 解析失败 {name}: {e}")
            return None

    @staticmethod
    def _phones_of(sd: SentenceData | None) -> list[tuple[float, float, str]]:
        if sd is None:
            return []
        return [(float(a), float(b), str(m))
                for a, b, m in zip(sd.phones.xmins, sd.phones.xmaxs, sd.phones.marks)]

    def _seed_tiers(self, data: SentenceData) -> None:
        """标准工程：工作文件里没有 words/texts 时，从源 TextGrid 播种这两层。

        `words_ref`（word 名自动更新的基线）按**源 phones + 源 words** 预先算好 ——
        只要音素没变，源 words 标记就原样保留；音素改过则首次刷新时自动改成音素集合。
        """
        if self.project is None or self.project.is_refurbish or self.tg is None:
            return
        if not data.words:
            data.words = tier_items(self.tg.words)
            data.texts = tier_items(self.tg.texts)
        elif not data.texts:
            data.texts = tier_items(self.tg.texts)
        if len(data.words_ref) != len(data.words):
            src_phones = self._phones_of(self.tg)
            data.words_ref = [[src_phones[i][2] for i in g]
                              for g in word_phone_groups(src_phones, data.words)]

    def _load_f0(self, name: str):
        path = self.project.f0_dir / f"{name}.npz"
        if not path.exists():
            self.view.set_f0(None)
            return
        try:
            f0, _uv, hop = load_f0(path)
            self.view.set_f0(f0, hop)
        except Exception as e:
            self._log(f"F0 缓存损坏 {name}: {e}")
            self.view.set_f0(None)

    # ================= 顶部层信息栏 =================
    def _update_label(self):
        self._update_tier_bar()

    def _update_tier_bar(self):
        if self.editor is None:
            for r in self.tier_bar.rows:
                r.items = []
            self.tier_bar.update()
            return
        if self.project is not None and self.project.is_refurbish:
            # 翻新工程只有 phones 层（层信息条也只挂这一行）
            self.tier_bar.set_phones(self.editor.phones)
            self.tier_bar.set_sel_phone(self.state.sel_phone)
            self.tier_bar.update()
            return
        # 标准工程：三层都取自编辑器（words/texts 可人工编辑，不再由 phones 反推）
        for name in ("texts", "words"):
            items = self.editor.texts if name == "texts" else self.editor.words
            self.tier_bar.set_tier(name, [a for a, _b, _m in items],
                                   [b for _a, b, _m in items],
                                   [m for _a, _b, m in items])
        self.tier_bar.set_phones(self.editor.phones)
        self.tier_bar.set_sel_phone(self.state.sel_phone)
        self.tier_bar.update()

    def _on_tier_phone_selected(self, idx: int):
        # 选中状态（含清掉 words/texts 选中）由层信息栏自己写入，这里只刷新视图与状态栏
        for v in (self.view, self.spec_view, self.wave):
            v.update()
        if 0 <= idx < len(self.editor.phones):
            s, e, m = self.editor.phones[idx]
            self.statusBar().showMessage(f"选中音素 [{idx + 1}] {m or '·'} {s:.3f}–{e:.3f}")

    def _on_tier_item_selected(self, tier: str, idx: int):
        """层信息栏里选中 words / texts 条目 → 刷新视图与状态栏
        （Delete / Ctrl+D / Ctrl+R 跟着**最近一次**的选择走）。"""
        if self.editor is None:
            return
        self.view.setFocus()        # 让快捷键落到能处理它们的画布视图上（目标 = 选中的条目）
        for v in (self.view, self.spec_view, self.wave):
            v.update()
        items = self.editor.words if tier == "words" else self.editor.texts
        if 0 <= idx < len(items):
            s, e, m = items[idx]
            kind = "词" if tier == "words" else "短语"
            self.statusBar().showMessage(
                f"选中{kind} [{idx + 1}] {m or '·'} {s:.3f}–{e:.3f}（Ctrl+D 复制 / Ctrl+R 重命名 / Delete 删除）")

    def _on_view_selection(self):
        """画布上选了音素 / 音符 → 刷新层信息栏（words / texts 的选中已在画布按下时作废）。"""
        self._update_tier_bar()

    # ================= 编辑与自动保存 =================
    def _on_edited(self):
        if self.editor is not None:
            sync_word_marks(self.editor)  # word 名：音素集合变了才改写，否则保留原标记
        self._update_tier_bar()
        for v in (self.view, self.spec_view, self.wave):
            v.update()
        self._mark_list_item()
        self._autosave_timer.start()

    def _save_current(self):
        if self.editor is None or self.project is None:
            return
        if self.audio.playing:
            # 播放中不做磁盘 IO（音频卡顿根因之一），停止 400ms 后补存
            self._autosave_timer.start()
            return
        try:
            self.editor.data.save(self.project.sentence_work_file(self.editor.name))
            self.editor.dirty = False
        except Exception as e:
            self._log(f"自动保存失败 {self.editor.name}: {e}")

    def _mark_list_item(self):
        """编辑后把该句标为「已编辑」（✓ + 灰）；持久化等信息在切句/关闭时落盘。"""
        if self.project is not None and self._current:
            self.project.edited.add(self._current)
        item = self._list_item_named(self._current)
        if item is not None:
            item.setText(self._item_text(self._current))
            item.setForeground(Qt.gray)

    def _refresh_views(self):
        for v in (self.view, self.spec_view, self.wave, self.curve):
            v.update()

    def _undo(self):
        if self.editor and self.editor.undo():
            self._refresh_views()
            self._on_edited()

    def _redo(self):
        if self.editor and self.editor.redo():
            self._refresh_views()
            self._on_edited()

    # ================= 设置/视图 =================
    def _on_cmap_action(self):
        act = self.sender()
        self._set_cmap(act.data())

    def _set_cmap(self, name: str):
        if name not in CMAPS:
            return
        for v in (self.view, self.spec_view):
            v.set_cmap(name)
        for act in self._cmap_group.actions():
            if act.data() == name:
                act.setChecked(True)
        # 配色是**软件级**设置：无工程也记住（换工程不丢）
        if self.app_settings.spectrum_preset != name:
            self.app_settings.spectrum_preset = name
            try:
                save_app_settings(self.app_settings, self.app_layout)
            except Exception:
                pass

    def _fit_view(self):
        self.state.fit()
        self._refresh_views()

    def _zoom(self, factor: float):
        for v in (self.view, self.wave):
            self.state.zoom(factor, v.plot_w() / 2, v.plot_w())
        self._refresh_views()

    def _sync_viewport(self):
        """任一面板缩放/平移/改音高行高/拖动边界 → 其余面板同步刷新（消除割裂感）。

        注意：必须重读 phones 数据（不能只 update 重绘），否则拖动过程中
        层信息条会停留在旧边界上 —— 表现为「松手后才同步」。
        """
        for v in (self.view, self.spec_view, self.wave, self.curve):
            v.update()
        self._update_tier_bar()

    def _on_cursor_moved(self, t: float):
        """任一面板的鼠标光标时间变化 → 广播给所有面板，让竖直虚线落在同一竖线上。

        只重绘时间真的变了的面板（发信者自身值未变，不重复重绘）。
        """
        for v in (self.view, self.spec_view, self.wave, self.curve):
            v._apply_cursor_t(t)

    def _reset_panes(self):
        h = max(3, self.panes.height())
        n = self.panes.count()
        if n == 0:                      # 全部面板都关：没有面板可分配高度
            return
        if n >= 4:
            # 四面板（含口型）：按面板角色分权重，波形/口型窄、钢琴卷帘宽
            wmap = {"wave": 0.16, "roll": 0.42, "curve": 0.19, "spec": 0.23}
            vis = _panel_visibility(self.app_settings)
            order = list(self.app_settings.panel_order) if self.app_settings else list(PANE_KEYS)
            self.panes.setSizes([max(1, int(h * wmap[k]))
                                 for k in order if k in wmap and vis[k]])
        elif n == 3:
            # 三面板默认按「波形窄 / 钢琴卷帘宽 / 频谱可拉小」分配
            self.panes.setSizes([int(h * 0.22), int(h * 0.50), int(h * 0.28)])
        elif n == 2:
            # 关掉波形面板时的两面板布局：钢琴卷帘占大头
            self.panes.setSizes([int(h * 0.28), int(h * 0.72)])
        else:
            self.panes.setSizes([h])

    def _on_wave_toggle(self, checked: bool):
        """「音符波形」开关：钢琴卷帘里音符组下方的波形包络（**软件级**设置，换工程不丢）。"""
        self.view.set_wave(checked)
        if self.app_settings.wave_show != bool(checked):
            self.app_settings.wave_show = bool(checked)
            try:
                save_app_settings(self.app_settings, self.app_layout)
            except Exception:
                pass
        self.statusBar().showMessage(f"音符波形已{'开启' if checked else '关闭'}", 4000)

    def _on_wave_panel_toggle(self, checked: bool):
        """独立波形面板开关（保留原入口，转调通用实现）。"""
        self._on_panel_visibility("wave", checked)

    def _on_panel_visibility(self, key: str, checked: bool):
        """各面板显隐开关（「显示」菜单与 设置→「面板」页同一份**软件级**设置）。

        只把面板按新顺序重新挂进 / 摘出 QSplitter（不重建 widget），
        故已加载的频谱 / 音符 / 播放头都保留、时间轴也不变。
        """
        was = bool(getattr(self.app_settings, _PANEL_ATTR[key], True))
        setattr(self.app_settings, _PANEL_ATTR[key], bool(checked))   # 无工程也记住
        try:
            save_app_settings(self.app_settings, self.app_layout)
        except Exception:
            pass
        self._apply_panes()
        # 刚打开口型面板且当前句还没缓存 → 立即补算当前句
        if key == "curve" and checked and not was and self._current:
            self._load_curve(self._current)
        self.statusBar().showMessage(
            f"{_PANEL_TITLE[key]}已{'显示' if checked else '隐藏'}", 4000)

    def _sync_panel_actions(self):
        """「显示」菜单勾选态跟着**软件级**设置走（启动即生效，无工程也同步）。"""
        st = self.app_settings
        for k, act in (("wave", self.act_wave_panel), ("roll", self.act_roll_panel),
                       ("curve", self.act_curve_panel), ("spec", self.act_spec_panel)):
            act.setChecked(bool(getattr(st, _PANEL_ATTR[k], True)))
        self.act_curve_fit.setChecked(bool(getattr(st, "curve_autoscale", True)))

    def _on_curve_autoscale(self, checked: bool):
        """「显示 → 口型按峰值自适应高度」：与 设置→「面板」页里那一项是同一份设置。"""
        self.app_settings.curve_autoscale = bool(checked)
        try:
            save_app_settings(self.app_settings, self.app_layout)
        except Exception:
            pass
        self.curve.set_autoscale(checked)
        self.statusBar().showMessage(
            "口型纵轴：按当前句峰值自适应高度" if checked
            else "口型纵轴：绝对标尺（0 = 闭嘴 / 1 = 最大张口）", 4000)

    def _apply_curve_options(self):
        """把**软件级**设置里口型面板的显示选项（纵轴自适应）推给口型画布。"""
        st = self.app_settings
        self.curve.set_autoscale(bool(st.curve_autoscale) if st is not None else True)

    def _apply_panes(self):
        """按设置的**上下顺序**重建面板的挂载（默认 波形 → 钢琴卷帘 → 口型 → 频谱）。

        - 顺序来自 ProjectSettings.panel_order（设置 →「面板」页可自由调整，由上到下）；
        - 各面板都有显隐开关（wave_panel / roll_panel / curve_panel / spec_panel，
          「显示」菜单同源），关掉的面板自动从顺序里跳过；全关时面板区为空
          （都要能关，不做强制兜底）；
        - 不重建 widget：只把面板按新顺序挂回 / 摘出 QSplitter（摘除用 setParent(None)），
          所以已加载的 spec / model / f0 / 播放头都保留，时间轴也不变。
        """
        st = self.app_settings                      # 软件级设置（无工程也生效）
        order = list(st.panel_order) if st is not None else list(PANE_KEYS)
        vis = _panel_visibility(st)
        widgets = {"wave": self.wave, "roll": self.view,
                   "curve": self.curve, "spec": self.spec_view}
        shown = [k for k in order if k in widgets and vis[k]]
        # 先全部摘下，再按新顺序挂回（addWidget 只会追加，顺序不对会导致面板错位）
        for w in widgets.values():
            if w.parent() is self.panes:
                w.setParent(None)
        for k in shown:
            self.panes.addWidget(widgets[k])
        for k, w in widgets.items():
            w.setVisible(k in shown)
        self._reset_panes()
        self._refresh_views()
        self._sync_viewport()

    def _capture_layout(self):
        """记下当前布局（面板长宽 + 钢琴窗行高与纵中心）。

        Write进 **软件级** `self.app_layout`（app_settings.json 的 layout 段）——
        面板参数跟着软件走，换工程不丢；无工程时也记录（下次启动照此恢复）。
        """
        self.app_layout.update({
            "main_sizes": [int(v) for v in self.main_split.sizes()],
            "panes_sizes": [int(v) for v in self.panes.sizes()],
            "panes_count": int(self.panes.count()),
            "piano_row_h": float(self.view.KEY_H),
            "piano_center": float(self.view.pitch_center),
        })

    def _restore_layout(self):
        """恢复软件级记忆的布局；面板数不匹配（本次改了波形面板显隐）→ 退回默认分配。"""
        lay = dict(self.app_layout or {})
        ps = lay.get("panes_sizes")
        if isinstance(ps, (list, tuple)) and len(ps) == self.panes.count():
            self.panes.setSizes([max(1, int(v)) for v in ps])
        else:
            self._reset_panes()
        ms = lay.get("main_sizes")
        if isinstance(ms, (list, tuple)) and len(ms) == self.main_split.count():
            self.main_split.setSizes([max(1, int(v)) for v in ms])
        self.view.set_pitch_view(row_h=lay.get("piano_row_h"),
                                 center=lay.get("piano_center"))

    def _about(self):
        QMessageBox.information(
            self, "DS-Labeler 快捷键",
            "空格：播放·停止　Z：适配全曲　Ctrl+S：保存工程\n"
            "中键拖动：平移画布（钢琴窗竖直方向改音高中心）\n"
            "滚轮：上下滚动　Shift+滚轮：水平滚动　Ctrl+滚轮：时间轴缩放　Alt+滚轮：音高行高缩放\n"
            "音素：拖动边界（自动夹取、不穿越）；双击重命名；Ctrl+D 复制；Delete 删除；Ctrl+R 重命名\n"
            "words / texts（层信息栏内）：先点中条目，再 Ctrl+D 复制 / Ctrl+R 重命名 / Delete 删除；"
            "左键拖边界（自动吸附下层边界）、双击改名\n"
            "音符：拖动改音高（自由）；拖两端改时间（吸附音素边界）；双击切分；↑↓ ±1 半音，Shift+↑↓ ±10 音分\n"
            "Ctrl+Z 撤销 / Ctrl+Y、Ctrl+Shift+Z 重做")

    # ================= 后台推理回调 =================
    def _on_infer_progress(self, name: str, done: int, total: int):
        self.statusBar().showMessage(f"预处理 {name}（{done}/{total}）")

    def _on_sentence_ready(self, name: str, data: SentenceData):
        # 注意：预处理完成**不**标记 ✓（那是「人工编辑过」的标记）；进度看状态栏即可
        if name != self._current:
            return
        # 仅当当前句尚无音符且未被用户编辑时才就地替换编辑器（避免覆盖用户改动）。
        # 注意守卫只挡「换编辑器」：F0 与编辑无关，当前句永远要刷新 PIT——
        # 翻新工程从 csv 建的编辑器天生带 notes，不拆开守卫会连 PIT 一起挡掉。
        if self.editor is None or not (self.editor.dirty or self.editor.notes):
            self.editor = EditModel(data)
            for v in (self.view, self.spec_view, self.wave, self.curve):
                v.set_model(self.editor)
            self._update_label()
        self._load_f0(name)

    def _on_infer_failed(self, name: str, err: str):
        self._log(f"推理失败 {name}: {err}")
        self.statusBar().showMessage(f"推理失败 {name}: {err}")

    def _on_provider(self, provider: str):
        self.statusBar().showMessage(f"推理设备：{provider}")

    def _on_infer_idle(self):
        self.statusBar().showMessage("后台预处理完成")
        self._log("后台预处理完成")

    def _batch_infer(self):
        if self.project is None or self.worker is None:
            return
        n = self.worker.enqueue([s for s in self._sentences if not self._is_ready(s)])
        self.statusBar().showMessage(f"已加入预处理队列：{n} 句")

    def _rerun_current(self):
        """重跑当前句：标准 → 丢弃音符与 F0 并重新 GAME 推理；翻新 → 只重算 F0。"""
        if self.project is None or self.worker is None or not self._current:
            return
        name = self._current
        what = "F0 缓存" if self.project.is_refurbish else "音符与 F0 缓存"
        ret = QMessageBox.question(
            self, "重跑当前句", f"将丢弃 {name} 的{what}并重新推理，继续？")
        if ret != QMessageBox.Yes:
            return
        try:
            if not self.project.is_refurbish:      # 翻新的音符来自 csv，无可丢弃的工作文件
                (self.project.sentence_work_file(name)).unlink(missing_ok=True)
            (self.project.f0_dir / f"{name}.npz").unlink(missing_ok=True)
        except Exception as e:
            self._log(f"删除缓存失败 {name}: {e}")
        self._load_sentence(name)
        self.worker.enqueue([name], front=True, skip_cached=False)

    # ================= 批量替换音素 =================
    def _batch_replace(self):
        if self.project is None:
            return
        old, ok = QInputDialog.getText(self, "批量替换音素", "要被替换的音素（旧）：")
        if not ok or not old:
            return
        new, ok = QInputDialog.getText(self, "批量替换音素", f"替换为（新）：")
        if not ok:
            return
        scope = QMessageBox.question(
            self, "替换范围", "替换整个工程？\n（否 = 只替换当前句）")
        whole = scope == QMessageBox.Yes
        names = self._sentences if whole else [self._current]
        changed = 0
        for name in names:
            if not name:
                continue
            data = (self.editor.data if (not whole and self.editor and name == self._current)
                    else self._read_work(name))
            if data is None:               # 未编辑句 → 翻新工程回退到 csv 源数据
                data = self.project.source_sentence(name)
            if data is None:
                continue
            n = 0
            for i, (s, e, m) in enumerate(data.phones):
                if m == old:
                    data.phones[i] = (s, e, new)
                    n += 1
            if n:
                changed += n
                data.save(self.project.sentence_work_file(name))
        if not whole and self.editor is not None and self._current:
            for v in (self.view, self.spec_view, self.wave, self.curve):
                v.set_model(self.editor)
        self._log(f"批量替换音素 {old} → {new}：共 {changed} 处")
        QMessageBox.information(self, "完成", f"共替换 {changed} 处。")

    # ================= 导出 =================
    def _export(self):
        """标准工程：导出数据集 → `<所选文件夹>/<音色名>/raw/{wavs/*.wav, transcriptions.csv}`。

        音色名由弹窗给（**必填**，预填上次用的名字；改了就记回工程，下次不用重写）。
        """
        if self.project is None:
            return
        if self.project.is_refurbish:      # 菜单项已按类型禁用；兜底转到「导出 CSV」
            self._export_csv()
            return
        voice, ok = QInputDialog.getText(
            self, "导出数据集",
            f"音色名称（导出到 <所选文件夹>/<音色名称>/raw/）：",
            text=self.project.voice_name or self.project.name)
        if not ok:
            return
        voice = _safe_name(voice)
        if not voice:
            QMessageBox.warning(self, "音色名称必填", "音色名称不能为空，导出已取消。")
            return
        folder = QFileDialog.getExistingDirectory(
            self, f"选择导出文件夹（将生成 {voice}/raw/…）",
            str(self.project.root))       # 默认停在**工程所在文件夹**（两类工程一致）
        if not folder:
            return
        out = pathlib.Path(folder) / voice / "raw"
        if self.project.voice_name != voice:      # 记住音色名（下次导出预填）
            self.project.voice_name = voice
            self.project.save()
        csv_path = out / "transcriptions.csv"
        if csv_path.exists() or (out / _EXPORT_AUDIO_DIR).exists():
            ret = QMessageBox.question(
                self, "覆盖导出", f"{out} 下已存在导出内容，将覆盖，继续？")
            if ret != QMessageBox.Yes:
                return
        try:
            self._save_current()
            classifier = self.project.make_classifier()
            items = []
            word_sizes = {}
            for name in self._sentences:
                data = (self.editor.data if (self.editor and name == self._current)
                        else self._read_work(name))
                if data is None:
                    continue
                items.append(data)
                # 词划分直接取自工作数据里**编辑后的 words 层**（不再由 phones 反推）
                ws = word_phone_counts(data.phones, data.words)
                if ws:
                    word_sizes[name] = ws
            unknown = unknown_report(items, classifier)
            if unknown:
                ret = QMessageBox.question(
                    self, "未命中音素",
                    "以下音素不在元辅表中（可能与训练配置不一致）：\n"
                    + ", ".join(unknown) + "\n\n仍继续导出？")
                if ret != QMessageBox.Yes:
                    return
            # 词/音符切分的软提示一律不再展示（MIDI 怎么改属于标注者自己的事，
            # 不该由工具事后点评）；硬校验仍在 export_dataset 内部执行，不合规照样拒写。
            csv_path = export_dataset(out, items, classifier, self.project.wav_dir,
                                      word_sizes_by_name=word_sizes)
            self._log(f"导出 {len(items)} 句 → {csv_path}")
            QMessageBox.information(
                self, "导出完成",
                f"已导出 {len(items)} 句：\n{csv_path}\n{(out / _EXPORT_AUDIO_DIR)}")
        except ExportValidationError as e:
            detail = []
            for nm, probs in list(e.problems.items())[:8]:
                detail.append(f"● {nm}：\n   - " + "\n   - ".join(probs))
            more = f"\n…另有 {len(e.problems) - 8} 句同类问题" if len(e.problems) > 8 else ""
            self._log(f"导出被拒：{len(e.problems)} 句不符合下游规则")
            QMessageBox.critical(
                self, "导出被拒（数据不合规）",
                "下列句子导出后无法通过 GAME/convert_ds 校验，已拒绝写出任何文件。\n"
                "请先在编辑器里修好再导出：\n\n" + "\n".join(detail) + more)
        except Exception as e:
            self._log(f"导出失败：{e}")
            QMessageBox.critical(self, "导出失败", str(e))

    def _export_csv(self):
        """翻新工程：把修正后的数据导出为一个新 csv（默认 `<工程名>_fixed.csv`）。

        对话框默认停在**工程所在文件夹**，文件名默认 `<工程名>_fixed.csv`；要用新 csv
        替换数据集，由用户在对话框里自行选到数据目录（属显式操作，不替用户做决定）。
        写盘前逐句按 convert_ds 硬规则校验，不过则拒绝写出。
        """
        if self.project is None or not self.project.is_refurbish:
            return
        default = str(self.project.root / f"{self.project.name}_fixed.csv")
        path, _ = QFileDialog.getSaveFileName(
            self, "导出 CSV（可自行选到数据目录以替换原 csv）", default,
            "CSV 文件 (*.csv);;所有文件 (*)")
        if not path:
            return
        out = pathlib.Path(path)
        if out.exists():
            ret = QMessageBox.question(self, "覆盖导出", f"{out} 已存在，将覆盖，继续？")
            if ret != QMessageBox.Yes:
                return
        try:
            self._save_current()
            classifier = self.project.make_classifier()
            items = []
            for name in self._sentences:
                data = (self.editor.data if (self.editor and name == self._current)
                        else self._read_work(name))
                if data is None:            # 未编辑句 → 用 csv 原数据
                    data = self.project.source_sentence(name)
                if data is not None:
                    items.append(data)
            csv_path = write_csv(out, items, classifier)
            self._log(f"导出 CSV {len(items)} 句 → {csv_path}")
            QMessageBox.information(
                self, "导出完成", f"已导出 {len(items)} 句：\n{csv_path}")
        except ExportValidationError as e:
            detail = []
            for nm, probs in list(e.problems.items())[:8]:
                detail.append(f"● {nm}：\n   - " + "\n   - ".join(probs))
            more = f"\n…另有 {len(e.problems) - 8} 句同类问题" if len(e.problems) > 8 else ""
            self._log(f"导出被拒：{len(e.problems)} 句不符合下游规则")
            QMessageBox.critical(
                self, "导出被拒（数据不合规）",
                "下列句子导出后无法通过下游校验，已拒绝写出任何文件。\n"
                "请先在编辑器里修好再导出：\n\n" + "\n".join(detail) + more)
        except Exception as e:
            self._log(f"导出 CSV 失败：{e}")
            QMessageBox.critical(self, "导出失败", str(e))

    # ================= 保存工程（Ctrl+S） =================
    def _save_project(self):
        """把当前句编辑落盘，并把工程进度（已编辑 / 停留句）+ 软件级设置与布局一起写回。

        工程侧车与软件级设置平时已随编辑自动落盘（见 _on_edited / _auto_save_layout），
        Ctrl+S 给的是一次「显式、立刻、可见反馈」的全量保存。
        """
        if self.project is None:
            self.statusBar().showMessage("没有打开工程，无法保存", 4000)
            return
        try:
            self._save_current()
            self._capture_layout()
            self.project.save()          # 同时写 app_settings.json（软件级设置 / 布局）
        except Exception as e:
            self._log(f"保存工程失败: {e}")
            QMessageBox.critical(self, "保存失败", str(e))
            return
        self._log(f"保存工程 {self.project.name}")
        self.statusBar().showMessage(f"已保存工程「{self.project.name}」", 4000)

    # ================= 回写 TextGrid =================
    def _writeback_tg(self):
        if self.project is None or self.editor is None:
            return
        if self.project.is_refurbish:
            QMessageBox.information(self, "不适用", "翻新工程没有 TextGrid，请用「导出 CSV」。")
            return
        name = self._current
        ret = QMessageBox.question(
            self, "回写 TextGrid",
            f"将用当前编辑结果覆盖 {name}.TextGrid？\n"
            "（phones / words / texts 三层都按编辑结果写出。此操作会修改源文件）")
        if ret != QMessageBox.Yes:
            return
        try:
            self._save_current()
            sync_word_marks(self.editor)          # 保证 word 名与音素集合一致
            phones = self.editor.phones
            from core.textgrid_io import write_edited_textgrid
            path = self.project.tg_dir / f"{name}.TextGrid"
            write_edited_textgrid(path, phones, self.editor.words, self.editor.texts)
            self._log(f"回写 TextGrid {name}")
            QMessageBox.information(self, "完成", f"已回写 {path}")
        except Exception as e:
            self._log(f"回写失败 {name}: {e}")
            QMessageBox.critical(self, "回写失败", str(e))

    # ================= 播放 =================
    def _toggle_play(self):
        if self.audio.playing:
            self.audio.stop()
            return
        t0, t1 = self.state.t0, self.state.t0 + self.state.span
        self.audio.play_range(t0, t1)
        self._timer.start()

    def _play_phoneme(self, t0: float, t1: float):
        """右键播放所点音素：只播 [start, end)，播放头定位到起点。"""
        self.audio.stop()
        self.audio.play_range(t0, t1)
        for v in (self.view, self.spec_view, self.wave, self.curve):
            v.set_play_pos(t0)
        self._timer.start()

    def _refresh_play_pos(self):
        pos = self.audio.current_pos()
        if pos < 0:
            return      # 停止后保留播放头在最后位置（播放指针必须一直可见）
        for v in (self.view, self.spec_view, self.wave, self.curve):
            v.set_play_pos(pos)

    # ================= 日志 =================
    def _log(self, msg: str):
        if self.project is not None:
            self._log_static(self.project, msg)

    @staticmethod
    def _log_static(project: Project, msg: str):
        try:
            path = project.work_dir / "run.log"
            path.parent.mkdir(parents=True, exist_ok=True)
            ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            with open(path, "a", encoding="utf8") as f:
                f.write(f"[{ts}] {msg}\n")
        except Exception:
            pass

    def closeEvent(self, event):
        self._save_current()
        try:
            self._capture_layout()      # 关闭时记住面板长宽 / 钢琴窗纵高
            if self.project is not None:
                self.project.save()     # （同时会把软件级设置与布局写回 app_settings.json）
            else:
                save_app_settings(self.app_settings, self.app_layout)
        except Exception:
            pass
        self.audio.stop()
        if self.worker is not None:
            self.worker.stop()
            self.worker.wait(15000)     # 预热/建会话可能正在跑（不可中断）：给足时间再析构 QThread
        if self.mouth_worker is not None:
            self.mouth_worker.stop()
            self.mouth_worker.wait(15000)   # 同上：口型推理不可中断，等它收尾
        super().closeEvent(event)
