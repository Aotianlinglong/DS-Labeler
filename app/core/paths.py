# -*- coding: utf-8 -*-
"""工程内资源路径（模型、词典、GAME 引擎）—— 单一事实源。"""
from __future__ import annotations

import os
import pathlib

# core/paths.py → parents: [0]=core, [1]=app, [2]=DS-Labeler 根
ROOT = pathlib.Path(__file__).resolve().parents[2]
APP_DIR = ROOT / "app"
ASSETS_DIR = ROOT / "assets"
ENGINE_DIR = ROOT / "engine"
MODELS_DIR = ROOT / "models"

# 软件级设置（跨工程共用，放软件文件夹根目录）：
# 界面设置（ProjectSettings：波形 / PIT / 配色 / 面板顺序 / 频谱参数）+ 面板布局记忆
# 默认落软件根目录；可用环境变量 DSL_APP_SETTINGS 改到别处（回归测试隔离用）。
APP_SETTINGS_FILE = pathlib.Path(
    os.environ.get("DSL_APP_SETTINGS") or (ROOT / "app_settings.json"))

# 翻新工程（旧数据集，csv + wavs）的**私有存储根**：工程文件与 F0 缓存都落这里，
# 绝不往数据目录里写任何东西（那是上游流水线的领地）。
# 默认落软件根目录 `refurbish/`；可用环境变量 DSL_REFURBISH_HOME 改到别处（回归测试隔离用）。
REFURBISH_DIR = pathlib.Path(
    os.environ.get("DSL_REFURBISH_HOME") or (ROOT / "refurbish"))

# 推理模型
RMVPE_MODEL = MODELS_DIR / "rmvpe" / "rmvpe.onnx"
GAME_MODEL_DIR = ENGINE_DIR / "GAME-ONNX" / "GAME-1.0.3-medium-onnx"

# 内置元辅词典
DICT_DIR = ASSETS_DIR / "dictionaries"


def lang_dict(lang: str) -> tuple[pathlib.Path, pathlib.Path, pathlib.Path]:
    """某语言内置词典的 (元音, 辅音, 流音) txt 路径（文件可能不存在 = 该表为空）。

    **四个语言（zh / ja / en / yue）的元辅表与英文流音都在这里**，是音素分类的
    唯一事实源；用户可直接编辑 / 增删这些 txt（见 README「音素词典」一节）。
    """
    d = DICT_DIR / lang
    return d / "vowels.txt", d / "consonants.txt", d / "liquids.txt"
