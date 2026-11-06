# Dataset Labeler（DS-Labeler）

面向 DiffSinger 数据集的标注工具：**TextGrid 音素层（texts / words / phones）+ 音高（MIDI 音符）**一体化标注。

读入 `wav/` + `TextGrid/`，自动生成音符与 F0，人工修正音素边界、词 / 短语与音符，最后导出 `raw/wavs + transcriptions.csv`，或回写 TextGrid。

## 功能

- **自动推理**：RMVPE 提取 F0、GAME 生成音符、R3MOE 提取口型曲线（ONNX Runtime，Windows 默认走 DirectML GPU）
- **句子列表**：质量分与五项诊断指标、表头排序、右键勾选列
- **音素搜索**：多音素任一命中；`[z ai]` 按顺序连续匹配；命中时段在面板常亮
- **精细编辑**：音素、words / texts、音符均可编辑；层间边界向上联动；切分音符自动对齐音高；全程撤销 / 重做
- **两类工程**：标准 TextGrid 工程，与只有 `transcriptions.csv` + 音频的翻新工程

## 基础编辑操作

- **播放**：`空格` 播放 / 停止；在频谱或钢琴卷帘右键可播放所点音素
- **音素层**：双击或 `Ctrl+R` 重命名、`Ctrl+D` 复制、`Delete` 删除；拖动边界不会穿越相邻音素，齐平的音符端点跟随
- **words / texts**：在层信息栏双击改名、拖动条目边界；选中条目后 `Ctrl+R` / `Ctrl+D` / `Delete`
- **音符**：拖动改音高、拖两端改时间、双击切分（切分时自动对齐音高）
- **视图**：滚轮上下、`Shift+滚轮` 水平、`Ctrl+滚轮` 缩放、`Alt+滚轮` 音高缩放；中键平移，`Z` 适配全曲
- **撤销 / 重做**：`Ctrl+Z` / `Ctrl+Y`

## 快速开始

需要 **Python ≥ 3.10**（已在 3.12 上验证）。

```bash
pip install -r requirements.txt
```

| 依赖 | 用途 |
|---|---|
| `PySide6-Essentials` | 界面（Qt6；只要 Essentials） |
| `numpy` / `scipy` | 数值计算、信号处理 |
| `soundfile` / `sounddevice` | 音频读写与播放 |
| `librosa` | STFT / mel 频谱 |
| `onnxruntime-directml` | RMVPE / GAME 推理（Windows + DX12 显卡走 GPU）。纯 CPU 环境换成 `onnxruntime`，其余不变 |

> `onnxruntime` 必须在任何 Qt 绑定**之前**导入，否则会因 VC 运行时冲突报「DLL 初始化例程失败」。入口 `app/main.py` 已内置守卫，请勿打乱导入顺序。

双击 `run.bat` 启动（依次解析环境变量 `DSLABELER_PY`、同目录 `python_path.txt`、`PATH` 上的 `python`）。

## 模型下载

权重**不随源码分发**，下载后按下列路径放置即可：

| 文件 | 放置位置 | 下载 |
|---|---|---|
| `rmvpe.onnx`（约 362 MB） | `models/rmvpe/rmvpe.onnx` | [RVC 项目的 RMVPE ONNX 导出](https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/rmvpe.onnx) |
| `GAME-1.0.3-medium-onnx/`（约 190 MB） | `engine/GAME-ONNX/` | [GAME release v1.0.3](https://github.com/openvpi/GAME/releases/tag/v1.0.3) |
| `opec.onnx`（约 40 MB，可选） | `models/r3moe/opec.onnx` | [R3MOE](https://github.com/KakaruHayate/R3MOE)，需按上游 `export_onnx.py` 自行导出 |

`GAME-1.0.3-medium-onnx/` 下应有 `encoder.onnx`、`segmenter.onnx`、`estimator.onnx`、`dur2bd.onnx`、`bd2dur.onnx`、`config.json`。
缺失模型时软件仍可启动，只是自动生成音符 / F0 相关功能不可用；口型权重缺失只影响口型面板。

## 目录结构

```
DS-Labeler/
├── run.bat                 # 启动脚本
├── requirements.txt        # 运行依赖
├── app/                    # 软件代码
│   ├── main.py             #   入口（环境引导 + 导入顺序守卫 + 崩溃日志）
│   ├── core/               #   数据模型与算法
│   │   ├── project.py      #     工程（文件夹级，project.json；含翻新工程）
│   │   ├── textgrid_io.py  #     TextGrid 读写（自写 Praat 写出器）
│   │   ├── sentence.py     #     单句工作数据（<工作目录>/<句名>.json）
│   │   ├── csv_io.py       #     旧数据集 transcriptions.csv 读写（翻新工程，只读数据目录）
│   │   ├── notes.py        #     音符模型（MIDI 保留音分）
│   │   ├── phonemes.py     #     多语言元辅分类 + ph_num
│   │   ├── spectrogram.py  #     mel 频谱（整段缓存 + 配色）
│   │   ├── curve.py        #     口型（开口度）曲线：R3MOE ONNX 推理 + 缓存读写
│   │   ├── pitch.py        #     传统音高判定（F0 曲线查表 / YIN），供切分对齐
│   │   ├── edit.py         #     编辑操作 + 层间联动 + 标签计算
│   │   ├── undo.py         #     命令栈（命令式 / 快照式）
│   │   ├── export.py       #     导出 raw/wavs + transcriptions.csv
│   │   └── paths.py        #     资源路径单一事实源
│   ├── ui/
│   │   ├── theme.py        #     深色主题
│   │   ├── main_window.py  #     主窗口（菜单 / 句子列表与音素搜索 / 层信息栏 / 自动保存）
│   │   ├── views.py        #     波形 / 频谱 / 钢琴卷帘 / 口型四视图 + 交互
│   │   ├── settings_dialog.py #  统一设置面板（语言 / 频谱 / 波形 / 面板 / 配色）
│   │   ├── worker.py       #     后台推理线程（RMVPE + GAME + 口型）
│   │   └── audio.py        #     音频播放（播放当前视窗）
│   └── engine/
│       ├── onnx_util.py    #     ONNX 会话（DirectML 优先 / CPU 回退）
│       ├── f0.py           #     RMVPE F0 提取 + F0 缓存
│       └── game.py         #     GAME 音符生成（压缩时间轴映射）
├── models/rmvpe/           # RMVPE 权重（自备，见「模型下载」）
├── models/r3moe/           # R3MOE 口型权重（自备，见「模型下载」）
├── engine/GAME-ONNX/       # GAME ONNX 模型（自备，见「模型下载」）
└── assets/dictionaries/    # 内置音素词典（可直接编辑，见使用手册）
```

## 工程类型

由目录结构自动判定：

| 类型 | 数据目录 | 工程文件 |
|---|---|---|
| 标准工程 | `wav/` + `TextGrid/` | `<数据目录>/project.json` |
| 翻新工程 | `transcriptions.csv` + `wavs/` | `<软件根>/refurbish/<名>.refurbish.json` |

翻新工程只读取数据目录，**绝不写入**。

## 文档

- [使用手册](docs/manual.md)：工程类型、完整编辑操作、搜索 / 排序、导出、音符标签规则、音素词典

## 第三方组件与致谢

| 组件 | 用途 | 许可 |
|---|---|---|
| [GAME](https://github.com/openvpi/GAME)（openvpi） | 音符自动生成 | 代码开源；附带的 ONNX 模型为 **CC BY-NC-SA 4.0（非商业）** |
| [RMVPE](https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI) | F0 提取 | 见上游 |
| [R3MOE](https://github.com/KakaruHayate/R3MOE) | 口型（开口度）曲线 | 代码 **MPL-2.0**；权重需自行导出，不随源码分发 |
| [DiffSinger](https://github.com/openvpi/DiffSinger) / [MakeDiffSinger](https://github.com/openvpi/MakeDiffSinger) | 数据格式与流水线约定 | 见各自仓库 |
| [Qt for Python (PySide6)](https://doc.qt.io/qtforpython/) | 界面框架 | LGPLv3 |
| numpy / scipy / librosa / soundfile / sounddevice / onnxruntime | 数值、音频、推理 | BSD / ISC / MIT 等 |

**设计参考**：[VLabeler](https://github.com/sdercolin/vlabeler)（sdercolin，Apache-2.0）——界面布局与交互设计参考了它的部分方案；本软件是独立的 Python + Qt 实现，未复制其代码。

完整归属与许可全文见 [CREDITS.md](CREDITS.md)。

## 许可

本仓库源码采用 **Apache License 2.0**，版权归 **Aotianlinglong** 所有（见 [LICENSE](LICENSE) / [NOTICE](NOTICE)）。
GAME 的 ONNX 模型为 **CC BY-NC-SA 4.0（非商业）**，独立于本仓库许可。

## 反馈

Bug 与功能建议请提交到 [Issues](https://github.com/Aotianlinglong/DS-Labeler/issues)，附版本号、系统与复现步骤。
