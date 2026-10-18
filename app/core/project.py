# -*- coding: utf-8 -*-
"""
工程（自有格式）：两类工程共用**同一套 UI**，只有数据来源与文件落点不同。

一、标准工程（新数据）：工程 = 文件夹（数据集级），文件都放在工程文件夹里。
  工程文件夹/
  ├── project.json          # 工程配置（主语言/语言包/全局音素/uv-vocab）
  ├── ds_labeler_settings.json   # 工程侧车（语言表 + 编辑进度；界面设置不在这里）
  ├── wav/                  # 音频（引用原文件或用户自放）
  ├── TextGrid/             # 音素标注
  └── .dsl/                 # DS-Labeler 工作数据
      ├── <句名>.json       # 每句编辑数据（音素/音符）
      ├── f0/<句名>.npz     # F0 缓存（RMVPE 结果）
      └── curve/<句名>.npz  # 口型缓存（R3MOE 结果）
  导出数据集时输出 <所选文件夹>/<音色名>/raw/{wavs/*.wav, transcriptions.csv}（见 core/export.py）。

二、翻新工程（旧数据集）：数据目录是**上游流水线的领地**，只读它，绝不往里写任何文件；
  自己的工程文件与 F0 缓存落**软件根目录的 refurbish/**。
  软件根/refurbish/
  ├── <工程名>.refurbish.json    # 工程文件（记录数据目录地址 + 语言/进度）
  └── cache/<工程名>/
      ├── <句名>.json            # 每句编辑数据（人工改过的才落盘）
      ├── f0/<句名>.npz          # F0 缓存（RMVPE 结果）
      └── curve/<句名>.npz       # 口型缓存（R3MOE 结果）
  音符直接取 csv 的 note_seq/note_dur/note_slur（不做 GAME 推理）。

两类工程的类型由**目录结构**判定（`detect_kind`）：
  wav/ + TextGrid/                      → standard
  transcriptions.csv + wavs/(或 wav/)   → refurbish
"""
from __future__ import annotations

import json
import os
import pathlib
from dataclasses import dataclass, field

from core.paths import APP_SETTINGS_FILE, REFURBISH_DIR, lang_dict
from core.phonemes import (
    LANGS, DEFAULT_GLOBAL_PHONEMES, DEFAULT_SYSTEMS,
    SYSTEMS, SymbolTypeTable, LanguageTable, PhonemeClassifier,
)

PROJECT_FILE = "project.json"
SETTINGS_FILE = "ds_labeler_settings.json"   # 标准工程侧车（语言/已编辑句/停留句；**不含界面设置**）
WORK_DIR = ".dsl"
F0_DIR = "f0"
CURVE_DIR = "curve"     # 口型（R3MOE）缓存：<工作目录>/curve/<句名>.npz

KIND_STANDARD = "standard"      # 新数据（wav/ + TextGrid/）
KIND_REFURBISH = "refurbish"    # 翻新旧数据集（transcriptions.csv + wavs/）
REFURBISH_SUFFIX = ".refurbish.json"   # 翻新工程文件后缀（与标准 project.json 区分）
REFURBISH_CACHE = "cache"       # 翻新工程工作数据的子目录名


def detect_kind(root: pathlib.Path) -> str | None:
    """按**目录结构**判定工程类型（新建工程时用）。

    标准：`wav/` + `TextGrid/` 都在；
    翻新：`transcriptions.csv` + `wavs/`（或 `wav/`、根下 *.wav），允许 `raw/` 一层嵌套。
    都不满足 → None。
    """
    root = pathlib.Path(root)
    if (root / "wav").is_dir() and (root / "TextGrid").is_dir():
        return KIND_STANDARD
    for base in (root, root / "raw"):
        if not base.is_dir():
            continue
        if not (base / "transcriptions.csv").exists():
            continue
        for wd in (base / "wavs", base / "wav", base):
            if wd.is_dir() and any(wd.glob("*.wav")):
                return KIND_REFURBISH
    return None


def refurbish_project_path(name: str) -> pathlib.Path:
    return REFURBISH_DIR / f"{name}{REFURBISH_SUFFIX}"


def unique_refurbish_name(root: pathlib.Path, base: str) -> str:
    """翻新工程名去重：同名但指向**同一目录** → 复用；指向不同目录 → 加 `-2`/`-3`。"""
    want = os.path.normcase(str(pathlib.Path(root).resolve()))
    base = base or "project"
    nm, k = base, 2
    while True:
        cand = refurbish_project_path(nm)
        if not cand.exists():
            return nm
        try:
            other = json.loads(cand.read_text(encoding="utf8")) or {}
        except (OSError, ValueError):
            return nm                       # 坏文件 → 用原名覆盖
        if os.path.normcase(str(other.get("data_root") or "")) == want:
            return nm                       # 同一目录 → 复用已有工程
        nm, k = f"{base}-{k}", k + 1


def find_refurbish_project(root: pathlib.Path) -> "Project | None":
    """软件根 `refurbish/` 下已有指向该数据目录的工程 → 返回它（避免重复建）。"""
    if not REFURBISH_DIR.is_dir():
        return None
    want = os.path.normcase(str(pathlib.Path(root).resolve()))
    for f in sorted(REFURBISH_DIR.glob(f"*{REFURBISH_SUFFIX}")):
        try:
            data = json.loads(f.read_text(encoding="utf8")) or {}
        except (OSError, ValueError):
            continue
        if os.path.normcase(str(data.get("data_root") or "")) == want:
            try:
                return Project.load(f)
            except Exception:
                return None
    return None

DEFAULT_UV_VOCAB = ["AP", "SP", "br", "sil"]

PANE_KEYS = ("wave", "roll", "curve", "spec")   # 面板标识：波形 / 钢琴卷帘 / 口型 / 频谱

# ---- 配色（音符 / PIT / 两处波形 / 口型曲线）-------------------------------
# 界面颜色的**唯一权威默认值**在这里（views.py 的绘制默认值必须与之一致，回归有断言钉死）。
# 约定：只存 "#rrggbb" 的 RGB，**alpha 由视图角色固定**——
#   波形面板 = 不透明(255)、钢琴卷帘波形 = 半透明(150)、
#   音符填充 = 205 / 描边 = 220（描边由填充色向白混合 0.85 得到）、PIT = 205。
# 口型曲线是「描边 + 填充」两色。
BASE_COLOR_KEYS = ("note_color", "pit_color", "wave_panel_color", "wave_roll_color")
COLOR_KEYS = BASE_COLOR_KEYS + ("curve_line_color", "curve_fill_color")

DEFAULT_COLORS: dict[str, str] = {
    "note_color": "#50beff",        # 普通音符填充（= rgb 80,190,255）
    "pit_color": "#ffcd5a",         # PIT / F0 音高线（= rgb 255,205,90）
    "wave_panel_color": "#7fb3ff",  # 独立波形面板的波形（= rgb 127,179,255，蓝）
    # 钢琴卷帘波形：**深靛蓝**。刻意**不用中性灰白**——灰白与常见音频工具观感撞脸、
    # 辨识度低，故采用更有辨识度的深靛蓝。
    "wave_roll_color": "#4b5e93",   # 钢琴卷帘波形（= rgb 75,94,147）
    # 口型曲线：亮蓝描边 + 深靛填充。背景不参与自定义——面板背景沿用软件主题色。
    "curve_line_color": "#7fb3ff",  # 口型曲线描边（= rgb 127,179,255）
    "curve_fill_color": "#303c5e",  # 口型曲线填充（= rgb 48,60,94）
}

# 内置配色方案（下拉可选；方案 = 一组颜色的一次性预设）。
# 「经典（默认）」= DEFAULT_COLORS（改默认值即改它）。
# 注意：音符一律避开绿色系——绿色是「连音(slur)音符」的语义标记色，撞色会分不清。
COLOR_SCHEMES: dict[str, dict[str, str]] = {
    "经典（默认）": dict(DEFAULT_COLORS),
    "落日暖橙": {"note_color": "#ffa726", "pit_color": "#ff7043",
                 "wave_panel_color": "#ffca28", "wave_roll_color": "#d8c7ae",
                 "curve_line_color": "#ff8a4c", "curve_fill_color": "#4d3126"},
    "深海青蓝": {"note_color": "#26c6da", "pit_color": "#ffd54f",
                 "wave_panel_color": "#4fc3f7", "wave_roll_color": "#a9c3d0",
                 "curve_line_color": "#29b6f6", "curve_fill_color": "#243b4a"},
    "紫罗兰夜": {"note_color": "#9575cd", "pit_color": "#ce93d8",
                 "wave_panel_color": "#b39ddb", "wave_roll_color": "#c0b6d0",
                 "curve_line_color": "#b39ddb", "curve_fill_color": "#3a2f4d"},
    "樱花粉": {"note_color": "#f48fb1", "pit_color": "#ffcc80",
               "wave_panel_color": "#f8bbd0", "wave_roll_color": "#d8c8cf",
               "curve_line_color": "#f06292", "curve_fill_color": "#4a2e3a"},
}


def scheme_of(colors: dict[str, str]) -> str | None:
    """反查当前颜色对应的**内置**方案名（用于打开设置时预选下拉）；自定义则 None。"""
    for name, cols in COLOR_SCHEMES.items():
        if all(str(colors.get(k, "")).lower() == v.lower() for k, v in cols.items()):
            return name
    return None


def clean_presets(raw) -> dict[str, dict[str, str]]:
    """规范化**自定义配色预设**：{名称: {颜色键: #rrggbb}}。

    只保留颜色形如 #rrggbb（或 #rgb）的条目；内置方案名不允许占用（避免歧义）。
    **向后兼容**：老版本预设只存基础四色（note/pit/两处波形）→ 口型两色缺失时补内置
    默认，保证用户已有的自存预设升级后不丢。基础四色仍需齐全（残缺条目仍被拒）。
    """
    out: dict[str, dict[str, str]] = {}
    if not isinstance(raw, dict):
        return out
    for name, cols in raw.items():
        if not isinstance(name, str) or not name.strip() or name.strip() in COLOR_SCHEMES:
            continue
        if not isinstance(cols, dict):
            continue
        item = {}
        for k in BASE_COLOR_KEYS:
            v = str(cols.get(k) or "").strip()
            if v[:1] == "#" and len(v) in (4, 7):
                item[k] = v
        if len(item) != len(BASE_COLOR_KEYS):
            continue
        for k in COLOR_KEYS:                 # 口型两色：有则用，缺则补默认
            if k in item:
                continue
            v = str(cols.get(k) or "").strip()
            item[k] = v if (v[:1] == "#" and len(v) in (4, 7)) else DEFAULT_COLORS[k]
        out[name.strip()] = {k: item[k] for k in COLOR_KEYS}
    return out


def _clean_panel_order(raw) -> list[str]:
    """规范化面板顺序：只留合法键、去重，缺的按默认序补上（保证永远是 4 个键）。"""
    order = [k for k in (raw or []) if isinstance(k, str) and k in PANE_KEYS]
    for k in PANE_KEYS:            # 缺的按**默认序**（wave→roll→curve→spec）补在后面
        if k not in order:
            order.append(k)
    return order


@dataclass
class ProjectSettings:
    """界面/行为设置。

    **归属**：这是**软件级**设置（所有工程共用），存在软件根目录的 `app_settings.json`
    （见 `core.paths.APP_SETTINGS_FILE`），**不写进工程**。
    默认值为常用的一套（频谱 hop=55 / 15kHz / 5.0 / dB[-10,55] / viridis，
    波形带高 120px / 外扩 100ms，PIT 1.5px，经典配色，默认隐藏独立波形面板）。
    """
    wave_show: bool = True                # 音符波形：贴在每个音符正下方的波形包络
    wave_panel: bool = False              # 独立波形面板是否显示（波形已进钢琴卷帘 → 默认关掉）
    roll_panel: bool = True               # 钢琴卷帘面板是否显示（显示菜单 / 设置→面板页可关）
    curve_panel: bool = False             # 口型面板是否显示（默认关；需 R3MOE 权重在场才有内容）
    spec_panel: bool = True               # 频谱面板是否显示（同上；四个面板都可关）
    wave_h: float = 120.0                 # 钢琴窗波形上下最大宽度（像素，固定不随缩放变）
    wave_ext_ms: float = 100.0            # 钢琴窗波形每段首尾的横向外扩时长（毫秒，非像素）
    pit_width: float = 1.5                # PIT（F0）音高线粗细（px）：1.0=最细、2.0=最粗
    label_font_size: int = 7              # 钢琴卷帘标签字号（pt）：音符块内 / 词首 / 音素名共用
    # 音符块**内**是否写标签文字（默认关：PIT 曲线会盖住块内文字；词首块外侧那份照旧）
    note_label_inner: bool = False
    # ---- 配色（音符 / PIT / 两处波形 / 口型曲线）：存 "#rrggbb"；alpha 由视图角色固定 ----
    note_color: str = DEFAULT_COLORS["note_color"]             # 普通音符颜色
    pit_color: str = DEFAULT_COLORS["pit_color"]               # PIT 音高线颜色
    wave_panel_color: str = DEFAULT_COLORS["wave_panel_color"]  # 独立波形面板的波形颜色
    wave_roll_color: str = DEFAULT_COLORS["wave_roll_color"]   # 钢琴卷帘波形的颜色
    curve_line_color: str = DEFAULT_COLORS["curve_line_color"]  # 口型曲线描边颜色
    curve_fill_color: str = DEFAULT_COLORS["curve_fill_color"]  # 口型曲线填充颜色
    # 口型曲线纵轴是否**按当前句峰值自适应**（把该句最大张口顶到面板顶端；关掉则绝对标尺
    # 0 底 / 1 顶）。默认开——否则大多数句子的曲线都贴在底部一小条，看不出形状。
    curve_autoscale: bool = True
    # 用户自存的配色预设 {名称: {各色}}，与内置 COLOR_SCHEMES 一起出现在「配色」页下拉
    color_presets: dict[str, dict[str, str]] = field(default_factory=dict)
    # 面板的上下顺序（"wave"=波形面板 / "roll"=钢琴卷帘 / "curve"=口型 / "spec"=频谱），
    # 由上到下依次挂载；隐藏的面板（wave_panel / roll_panel / curve_panel / spec_panel
    # 为 False）自动跳过。默认：波形 → 钢琴卷帘 → 口型 → 频谱。
    panel_order: list[str] = field(
        default_factory=lambda: ["wave", "roll", "curve", "spec"])
    spectrum_preset: str = "viridis"      # 频谱配色（magma/viridis/gray；默认 viridis）
    # ---- 频谱自定义（窗长按采样率自适应，不暴露）----
    # 默认值为常用的一套
    spec_window: str = "blackmanharris"   # 窗函数
    spec_hop: int = 55                    # FFT 间隔
    spec_max_freq: float = 15000.0        # 最大显示频率（Hz）
    spec_mel_res: float = 5.0             # 频率分辨率（每行多少 mel；换频率轴时行数不变）
    spec_scale: str = "mel"               # 频率轴：mel / linear / log（见 spectrogram.SCALES）
    spec_db_min: float = -10.0            # 最小显示强度（dB）
    spec_db_max: float = 55.0             # 最大显示强度（dB）


def settings_to_dict(s: ProjectSettings) -> dict:
    """ProjectSettings → 可 JSON 化的 dict（**软件级**设置，落 app_settings.json）。"""
    return {
        "wave_show": s.wave_show,
        "wave_panel": s.wave_panel,
        "roll_panel": s.roll_panel,
        "curve_panel": s.curve_panel,
        "spec_panel": s.spec_panel,
        "wave_h": s.wave_h,
        "wave_ext_ms": s.wave_ext_ms,
        "pit_width": s.pit_width,
        "label_font_size": s.label_font_size,
        "note_label_inner": s.note_label_inner,
        "note_color": s.note_color,
        "pit_color": s.pit_color,
        "wave_panel_color": s.wave_panel_color,
        "wave_roll_color": s.wave_roll_color,
        "curve_line_color": s.curve_line_color,
        "curve_fill_color": s.curve_fill_color,
        "curve_autoscale": s.curve_autoscale,
        "color_presets": {k: dict(v) for k, v in s.color_presets.items()},
        "panel_order": list(s.panel_order),
        "spectrum_preset": s.spectrum_preset,
        "spec_window": s.spec_window,
        "spec_hop": s.spec_hop,
        "spec_max_freq": s.spec_max_freq,
        "spec_mel_res": s.spec_mel_res,
        "spec_scale": s.spec_scale,
        "spec_db_min": s.spec_db_min,
        "spec_db_max": s.spec_db_max,
    }


def settings_from_dict(st: dict | None) -> ProjectSettings:
    """dict → ProjectSettings（缺项走内置默认；容错脏数据）。"""
    st = st or {}
    d = ProjectSettings()
    return ProjectSettings(
        wave_show=bool(st.get("wave_show", d.wave_show)),
        wave_panel=bool(st.get("wave_panel", d.wave_panel)),
        roll_panel=bool(st.get("roll_panel", d.roll_panel)),
        curve_panel=bool(st.get("curve_panel", d.curve_panel)),
        spec_panel=bool(st.get("spec_panel", d.spec_panel)),
        wave_h=float(st.get("wave_h", d.wave_h) or d.wave_h),
        wave_ext_ms=float(st.get("wave_ext_ms", d.wave_ext_ms)),
        pit_width=float(st.get("pit_width", d.pit_width) or d.pit_width),
        label_font_size=int(st.get("label_font_size", d.label_font_size) or d.label_font_size),
        note_label_inner=bool(st.get("note_label_inner", d.note_label_inner)),
        note_color=str(st.get("note_color") or d.note_color),
        pit_color=str(st.get("pit_color") or d.pit_color),
        wave_panel_color=str(st.get("wave_panel_color") or d.wave_panel_color),
        wave_roll_color=str(st.get("wave_roll_color") or d.wave_roll_color),
        curve_line_color=str(st.get("curve_line_color") or d.curve_line_color),
        curve_fill_color=str(st.get("curve_fill_color") or d.curve_fill_color),
        curve_autoscale=bool(st.get("curve_autoscale", d.curve_autoscale)),
        color_presets=clean_presets(st.get("color_presets")),
        panel_order=_clean_panel_order(st.get("panel_order")),
        spectrum_preset=st.get("spectrum_preset") or d.spectrum_preset,
        spec_window=st.get("spec_window") or d.spec_window,
        spec_hop=int(st.get("spec_hop", d.spec_hop) or d.spec_hop),
        spec_max_freq=float(st.get("spec_max_freq", d.spec_max_freq) or d.spec_max_freq),
        spec_mel_res=float(st.get("spec_mel_res", d.spec_mel_res) or d.spec_mel_res),
        spec_scale=st.get("spec_scale") or d.spec_scale,
        spec_db_min=float(st.get("spec_db_min", d.spec_db_min)),
        spec_db_max=float(st.get("spec_db_max", d.spec_db_max)),
    )


def load_app_settings() -> tuple[ProjectSettings, dict]:
    """读**软件级**设置（app_settings.json）→ (ProjectSettings, layout)。

    文件不存在/损坏 → 内置默认（不报错，保证软件总能起来）。
    """
    if not APP_SETTINGS_FILE.exists():
        return ProjectSettings(), {}
    try:
        data = json.loads(APP_SETTINGS_FILE.read_text(encoding="utf8")) or {}
    except (OSError, ValueError):
        return ProjectSettings(), {}
    return settings_from_dict(data.get("settings")), dict(data.get("layout") or {})


def save_app_settings(settings: ProjectSettings, layout: dict | None = None) -> None:
    """写**软件级**设置到软件根目录（与任何工程无关，切换工程不影响）。"""
    try:
        APP_SETTINGS_FILE.parent.mkdir(parents=True, exist_ok=True)
        APP_SETTINGS_FILE.write_text(
            json.dumps({"version": 1,
                        "settings": settings_to_dict(settings),
                        "layout": dict(layout or {})},
                       ensure_ascii=False, indent=2),
            encoding="utf8")
    except OSError:
        pass          # 只读目录等场景：不因存设置失败而打断使用


@dataclass
class LanguageConfig:
    """每种语言的音素系统 + 词典/元辅流音表配置（可自定义）"""
    system: str = ""        # 音素系统（空 = 内置默认：zh/ja 二分式、yue 单音节、en 多音节）
    vowels: str = ""        # 元音 txt 路径（空 = 内置默认）
    consonants: str = ""    # 辅音 txt 路径
    liquids: str = ""       # 流音 txt 路径（多音节系统需要）
    dictionary: str = ""    # 词典 txt 路径（G2P / 二分式反推，可选）


@dataclass
class Project:
    """工程对象（内存态）"""
    root: pathlib.Path
    name: str = "未命名工程"
    main_lang: str = "zh"
    languages: dict[str, LanguageConfig] = field(default_factory=dict)
    symbol_table: str = ""       # 符号类型表（如训练配置 phones.txt，显式 symbol→类型）
    global_phonemes: list[str] = field(default_factory=lambda: list(DEFAULT_GLOBAL_PHONEMES))
    uv_vocab: list[str] = field(default_factory=lambda: list(DEFAULT_UV_VOCAB))
    settings: ProjectSettings = field(default_factory=ProjectSettings)
    edited: set[str] = field(default_factory=set)   # 已人工编辑过的句名（列表显示 ✓）
    last_sentence: str = ""                          # 上次停留的句名（重开工程时恢复）
    layout: dict = field(default_factory=dict)       # 关闭时的布局记忆
                                                     # （面板长宽 / 钢琴窗行高与纵中心）
    kind: str = KIND_STANDARD                        # "standard" | "refurbish"
    csv_rel: str = ""                                # 翻新：transcriptions.csv 相对 root 的路径
    wav_rel: str = ""                                # 翻新：wav 目录相对 root 的路径
    voice_name: str = ""                             # 导出数据集时的音色名（记在工程里，下次预填）

    # ---- 路径 ---------------------------------------------------------
    @property
    def is_refurbish(self) -> bool:
        return self.kind == KIND_REFURBISH

    @property
    def project_file(self) -> pathlib.Path:
        """工程文件：standard = `root/project.json`；refurbish = `ROOT/refurbish/<名>.refurbish.json`。"""
        if self.is_refurbish:
            return refurbish_project_path(self.name)
        return self.root / PROJECT_FILE

    @property
    def sidecar_file(self) -> pathlib.Path:
        """工程侧车。翻新工程只用**一个**工程文件（无独立侧车）。"""
        return self.project_file if self.is_refurbish else self.root / SETTINGS_FILE

    @property
    def wav_dir(self) -> pathlib.Path:
        if self.is_refurbish:
            return (self.root / self.wav_rel) if self.wav_rel else self.root
        return self.root / "wav"

    @property
    def tg_dir(self) -> pathlib.Path:
        return self.root / "TextGrid"

    @property
    def csv_file(self) -> pathlib.Path | None:
        """翻新工程的数据 csv（只读）。"""
        return (self.root / self.csv_rel) if (self.is_refurbish and self.csv_rel) else None

    @property
    def work_dir(self) -> pathlib.Path:
        if self.is_refurbish:
            return REFURBISH_DIR / REFURBISH_CACHE / self.name
        return self.root / WORK_DIR

    @property
    def f0_dir(self) -> pathlib.Path:
        return self.work_dir / F0_DIR

    @property
    def curve_dir(self) -> pathlib.Path:
        """口型（R3MOE）缓存目录：标准工程 `<根>/.dsl/curve/`，翻新 `<软件根>/refurbish/cache/<名>/curve/`。"""
        return self.work_dir / CURVE_DIR

    def wav_file(self, name: str) -> pathlib.Path:
        return self.wav_dir / f"{name}.wav"

    @property
    def diagnosis_file(self) -> pathlib.Path:
        """标注质量诊断（由上游对齐管线生成）。"""
        return self.tg_dir / "statistics" / "diagnosis.json"

    def load_diagnosis(self) -> dict[str, dict]:
        """读取每句标注质量指标（confidence/determinacy/monotonicity/agreement）。

        返回 {句名: 指标dict}；文件不存在或解析失败返回空 dict（不影响使用）。
        """
        try:
            data = json.loads(self.diagnosis_file.read_text(encoding="utf8"))
        except (OSError, ValueError):
            return {}
        out: dict[str, dict] = {}
        if isinstance(data, list):
            for item in data:
                if isinstance(item, dict) and "identifier" in item:
                    out[str(item["identifier"])] = item
        return out

    @staticmethod
    def diagnosis_score(metrics: dict | None) -> float | None:
        """把四项 0-1 指标合成一个质量分（算术平均），无指标返回 None。"""
        if not metrics:
            return None
        vals = [metrics.get(k) for k in
                ("confidence", "determinacy", "monotonicity", "agreement")]
        vals = [float(v) for v in vals if isinstance(v, (int, float))]
        return sum(vals) / len(vals) if vals else None

    # ---- 句子清单 -----------------------------------------------------
    def list_sentences(self) -> list[str]:
        """句名列表，按名称排序。

        标准工程：已配对（wav + TextGrid 同名）的句；翻新工程：数据 csv 的 name 列。
        """
        if self.is_refurbish:
            from core.csv_io import read_csv_names
            return read_csv_names(self.csv_file) if self.csv_file else []
        wavs = {p.stem for p in self.wav_dir.glob("*.wav")}
        tgs = {p.stem for p in self.tg_dir.glob("*.TextGrid")}
        names = sorted(wavs & tgs)
        return names

    def source_sentence(self, name: str) -> "SentenceData | None":
        """未编辑句的**源数据**：翻新工程从 csv 取该句（phones + notes）。

        标准工程返回 None —— 其源数据来自 TextGrid，由 MainWindow 自行解析
        （因为它还要保留 texts/words 三层供层信息栏参照）。
        """
        if not self.is_refurbish or self.csv_file is None:
            return None
        from core.csv_io import read_csv_items
        for it in read_csv_items(self.csv_file):
            if it.name == name:
                return it
        return None

    def sentence_work_file(self, name: str) -> pathlib.Path:
        return self.work_dir / f"{name}.json"

    # ---- 分类器 -------------------------------------------------------
    def system_of(self, lang: str) -> str:
        """该语言配置的音素系统（未配置 → 内置默认，再退化为二分式）。"""
        cfg = self.languages.get(lang)
        s = (cfg.system if cfg else "") or DEFAULT_SYSTEMS.get(lang, "binary")
        return s if s in SYSTEMS else "binary"

    @staticmethod
    def _load_lines(path: str) -> set:
        p = pathlib.Path(path)
        if not path or not p.exists():
            return set()
        with open(p, "r", encoding="utf8") as f:
            return set(f.read().split())

    @staticmethod
    def _dict_tables(dictionary: str) -> tuple[set, set]:
        """二分式词典反推（同 add_ph_num.py dictionary 分支）：每条 ≤2 音素。"""
        vowels, consonants = set(), set()
        p = pathlib.Path(dictionary)
        if not dictionary or not p.exists():
            return vowels, consonants
        for line in p.read_text(encoding="utf8").splitlines():
            parts = line.split("\t")
            if len(parts) < 2:
                continue
            phonemes = parts[1].split()
            if len(phonemes) == 1:
                vowels.add(phonemes[0])
            elif len(phonemes) == 2:
                consonants.add(phonemes[0])
                vowels.add(phonemes[1])
        return vowels, consonants

    def make_classifier(self) -> PhonemeClassifier:
        tables: dict[str, LanguageTable] = {}
        systems: dict[str, str] = {}
        for lang in LANGS:
            cfg = self.languages.get(lang, LanguageConfig())
            systems[lang] = self.system_of(lang)
            table = LanguageTable(lang)
            # 自定义 txt 优先
            table.vowels.update(self._load_lines(cfg.vowels))
            table.consonants.update(self._load_lines(cfg.consonants))
            table.liquids.update(self._load_lines(cfg.liquids))
            # 二分式词典反推（可选）
            if cfg.dictionary:
                dv, dc = self._dict_tables(cfg.dictionary)
                table.vowels.update(dv)
                table.consonants.update(dc)
            # 再次内置词典（assets/dictionaries/<lang>/；用户可直接编辑 / 增删音素）
            _vp, _cp, _lp = lang_dict(lang)
            if not table.liquids:
                table.liquids.update(self._load_lines(str(_lp)))
            if not table.vowels and not table.consonants:
                table.load_from_files(_vp, _cp)
            tables[lang] = table
        symbol_types = SymbolTypeTable.from_file(self.symbol_table)
        return PhonemeClassifier(
            main_lang=self.main_lang,
            tables=tables,
            global_phonemes=self.global_phonemes,
            systems=systems,
            symbol_types=symbol_types,
        )

    # ---- 保存/加载 ----------------------------------------------------
    def _lang_settings(self) -> dict:
        """工程侧车（`ds_labeler_settings.json`）：语言 + 编辑进度。

        **界面设置与布局不再写这里**（它们属软件级 → `app_settings.json`）。
        """
        return {
            "version": 1,
            "main_lang": self.main_lang,
            "symbol_table": self.symbol_table,
            "global_phonemes": self.global_phonemes,
            "edited_sentences": sorted(self.edited),
            "last_sentence": self.last_sentence,
            "voice_name": self.voice_name,
            "languages": {
                lang: {
                    "system": cfg.system,
                    "vowels": cfg.vowels,
                    "consonants": cfg.consonants,
                    "liquids": cfg.liquids,
                    "dictionary": cfg.dictionary,
                }
                for lang, cfg in self.languages.items()
            },
        }

    def to_dict(self) -> dict:
        """工程数据（**不含界面设置**：那些在软件根目录的 app_settings.json）。"""
        return {
            "name": self.name,
            "main_lang": self.main_lang,
            "uv_vocab": self.uv_vocab,
        }

    def _refurbish_dict(self) -> dict:
        """翻新工程文件内容：数据目录地址 + 语言/进度（单文件，无独立侧车）。"""
        d = self._lang_settings()
        d.update({
            "version": 1,
            "kind": KIND_REFURBISH,
            "name": self.name,
            "data_root": str(self.root),
            "csv_rel": self.csv_rel,
            "wav_rel": self.wav_rel,
            "uv_vocab": self.uv_vocab,
        })
        return d

    def save(self) -> None:
        self.f0_dir.mkdir(parents=True, exist_ok=True)      # 工作数据目录（含 f0/）
        if self.is_refurbish:
            # 翻新：**绝不碰数据目录** —— 工程文件落软件根 refurbish/<名>.refurbish.json
            self.project_file.parent.mkdir(parents=True, exist_ok=True)
            self.project_file.write_text(
                json.dumps(self._refurbish_dict(), ensure_ascii=False, indent=2),
                encoding="utf8",
            )
        else:
            self.root.mkdir(parents=True, exist_ok=True)
            (self.root / PROJECT_FILE).write_text(
                json.dumps(self.to_dict(), ensure_ascii=False, indent=2),
                encoding="utf8",
            )
            (self.root / SETTINGS_FILE).write_text(
                json.dumps(self._lang_settings(), ensure_ascii=False, indent=2),
                encoding="utf8",
            )
        # 界面设置 + 布局 → 软件根目录（跟着软件走，换工程不丢）
        save_app_settings(self.settings, self.layout)

    @staticmethod
    def _langs_from(data: dict) -> dict[str, LanguageConfig]:
        langs = {}
        for lang, cfg in (data.get("languages") or {}).items():
            langs[lang] = LanguageConfig(
                system=cfg.get("system", ""),
                vowels=cfg.get("vowels", ""),
                consonants=cfg.get("consonants", ""),
                liquids=cfg.get("liquids", ""),
                dictionary=cfg.get("dictionary", ""),
            )
        return langs

    @classmethod
    def load(cls, path: pathlib.Path) -> "Project":
        path = pathlib.Path(path)
        # 翻新工程文件：`<名>.refurbish.json`（内容含数据目录地址）
        if path.name.endswith(REFURBISH_SUFFIX):
            data = json.loads(path.read_text(encoding="utf8")) or {}
            root = pathlib.Path(str(data.get("data_root") or "."))
            csv_rel = str(data.get("csv_rel") or "")
            wav_rel = str(data.get("wav_rel") or "")
            if not csv_rel or not wav_rel:      # 手改过的文件 → 按目录重新探测
                from core.csv_io import detect_source
                c, w = detect_source(root)
                if c is not None and not csv_rel:
                    csv_rel = str(c.relative_to(root)).replace("\\", "/")
                if w is not None and not wav_rel:
                    wav_rel = str(w.relative_to(root)).replace("\\", "/")
            settings, layout = load_app_settings()
            return cls(
                root=root,
                name=data.get("name") or path.name[:-len(REFURBISH_SUFFIX)],
                kind=KIND_REFURBISH,
                csv_rel=csv_rel,
                wav_rel=wav_rel,
                main_lang=data.get("main_lang", ""),
                languages=cls._langs_from(data),
                symbol_table=data.get("symbol_table", ""),
                global_phonemes=data.get(
                    "global_phonemes", list(DEFAULT_GLOBAL_PHONEMES)),
                uv_vocab=data.get("uv_vocab", list(DEFAULT_UV_VOCAB)),
                # `edited` 是旧版工程文件（旧键名）的兼容读法
                edited=set(data.get("edited_sentences") or data.get("edited") or []),
                last_sentence=data.get("last_sentence", ""),
                voice_name=data.get("voice_name", ""),
                layout=layout,
                settings=settings,
            )
        root = path.parent if path.is_file() else path
        cfg_path = root / PROJECT_FILE
        if not cfg_path.exists():
            raise FileNotFoundError(f"不是有效的 DS-Labeler 工程：{root}")
        data = json.loads(cfg_path.read_text(encoding="utf8"))
        # 语言等设置优先从 SETTINGS_FILE 读；**主语言必填**：文件里没写就读成空串，
        # 由界面在打开前强制补选（绝不静默当 zh）。
        # 音素系统无记忆时用内置默认（zh/ja 二分式…）。
        side: dict = {}
        sp = root / SETTINGS_FILE
        if sp.exists():
            try:
                side = json.loads(sp.read_text(encoding="utf8")) or {}
            except (OSError, ValueError):
                side = {}
        # 界面设置 / 布局：**软件级**（app_settings.json）优先；
        # 文件还没生成时，回退读老工程里遗留的 project.json settings 段与侧车 layout
        # （一次性迁移：用户现存工程的自定义值不会丢）。
        if APP_SETTINGS_FILE.exists():
            settings, layout = load_app_settings()
        else:
            settings = settings_from_dict(data.get("settings"))
            layout = dict(side.get("layout") or {})
        return cls(
            root=root,
            name=data.get("name", "未命名工程"),
            main_lang=side.get("main_lang", data.get("main_lang", "")),
            languages=cls._langs_from(side if side else data),
            symbol_table=side.get("symbol_table", ""),
            global_phonemes=side.get(
                "global_phonemes", data.get("global_phonemes", list(DEFAULT_GLOBAL_PHONEMES))),
            uv_vocab=data.get("uv_vocab", list(DEFAULT_UV_VOCAB)),
            edited=set(side.get("edited_sentences", []) or []),
            last_sentence=side.get("last_sentence", ""),
            voice_name=side.get("voice_name", ""),
            layout=layout,
            settings=settings,
        )

    @classmethod
    def create(cls, root: pathlib.Path, name: str, main_lang: str = "zh",
               kind: str = KIND_STANDARD) -> "Project":
        """新建工程。kind 由调用方按**目录结构**判定（见 `detect_kind`）后传入。

        - standard：要求 root 下已有 wav/ 与 TextGrid/，文件都放工程文件夹里；
        - refurbish：要求 root 下有 transcriptions.csv + wavs/，工程文件与 F0 缓存落软件根
          `refurbish/`，**绝不往数据目录写任何东西**。
        界面设置沿用**软件级**当下的设置（app_settings.json）。
        """
        root = pathlib.Path(root)
        assert main_lang in LANGS, f"主语言必须是 {LANGS} 之一"
        settings, layout = load_app_settings()
        if kind == KIND_REFURBISH:
            from core.csv_io import detect_source
            csv_path, wav_dir = detect_source(root)
            if csv_path is None or wav_dir is None:
                raise ValueError("翻新工程需要目录内有 transcriptions.csv 与 wavs/（或 wav/）")
            p = cls(
                root=root, name=unique_refurbish_name(root, name),
                main_lang=main_lang, kind=KIND_REFURBISH,
                csv_rel=str(csv_path.relative_to(root)).replace("\\", "/"),
                wav_rel=str(wav_dir.relative_to(root)).replace("\\", "/"),
                settings=settings,
            )
        else:
            wav_dir, tg_dir = root / "wav", root / "TextGrid"
            if not wav_dir.is_dir() or not tg_dir.is_dir():
                raise ValueError("工程文件夹内需要已有 wav/ 与 TextGrid/ 两个子文件夹")
            p = cls(root=root, name=name, main_lang=main_lang,
                    kind=KIND_STANDARD, settings=settings)
        p.languages = {lang: LanguageConfig() for lang in LANGS}
        p.save()
        return p
