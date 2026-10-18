# Dataset Labeler（DS-Labeler）

**Dataset Labeler**（数据集标注器）为DiffSinger 数据集设计的标注工具 —— **TextGrid 音素层标注（texts / words / phones）+ 音高（MIDI / 音符）标注**
一体化桌面软件。

面向 DiffSinger 数据集制作流水线：读入 `wav/` + `TextGrid/`，用神经模型自动生成音符与 F0，
人工修正音素边界、词 / 短语层与音符，最后导出 `raw/wavs + transcriptions.csv`，或把修正结果回写 TextGrid。

- **交互**：三面板（波形 / mel 频谱 / 钢琴卷帘）同轴对齐，缩放与滚动同步
- **推理**：RMVPE 提取 F0、GAME 生成音符（ONNX Runtime，Windows 下默认走 DirectML GPU）
- **编辑**：音素层与词 / 短语层可独立编辑，边界对齐时向上联动；所有编辑均可撤销 / 重做（`Ctrl+Z` / `Ctrl+Y`）
- **两类工程**：标准 TextGrid 工程，以及只有 `transcriptions.csv` + 音频的「翻新工程」

## 目录

1. [运行](#1-运行)
2. [模型放置](#2-模型放置)
3. [目录结构](#3-目录结构)
4. [工程类型](#4-工程类型)
5. [使用](#5-使用)
6. [音符标签规则](#6-音符标签规则)
7. [音素词典（可直接编辑）](#7-音素词典可直接编辑)
8. [致谢与第三方](#8-致谢与第三方)
9. [许可](#9-许可)
10. [反馈与贡献](#10-反馈与贡献)

---

## 1. 运行

```
双击 run.bat
```

`run.bat` 按以下顺序解析 Python 解释器：

1. 环境变量 `DSLABELER_PY`（完整路径）
2. 与本脚本同目录的 `python_path.txt`（文件内容 = 一行解释器完整路径）
3. `PATH` 上的 `python`

要求 **Python ≥ 3.10**（已在 3.12 上验证）。首次运行前：

```
pip install -r requirements.txt
```

### 依赖说明（`requirements.txt`）

| 包 | 用途 |
|---|---|
| `PySide6-Essentials` | 界面框架（Qt6；只要 Essentials，不需要完整版） |
| `numpy` / `scipy` | 数值计算、信号处理 |
| `soundfile` / `sounddevice` | 音频读写与播放 |
| `librosa` | STFT / mel 频谱 |
| `onnxruntime-directml` | RMVPE / GAME 推理（Windows + DX12 显卡走 GPU）。**纯 CPU 环境把这一行换成 `onnxruntime`**，其余不变，速度明显下降 |

**关键点（务必保持）**：`onnxruntime` 必须在任何 Qt 绑定**之前**导入 —— 旧版 Qt 绑定的
VC 运行时与 `onnxruntime_pybind11_state` 冲突，后导入会报「DLL 初始化例程失败」。
入口脚本 `app/main.py` 已内置这条守卫，改入口时不要打乱导入顺序。

---

## 2. 模型放置

权重**不随源码分发**（体积大且各有自己的许可）。下载后按下表放好即可，目录名需一致：

| 文件 / 目录 | 放置位置 | 来源 |
|---|---|---|
| `rmvpe.onnx`（约 362 MB） | `models/rmvpe/rmvpe.onnx` | [RVC 项目](https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI) 使用的 RMVPE ONNX 导出：<https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/rmvpe.onnx> |
| `GAME-1.0.3-medium-onnx/`（6 个文件，约 190 MB） | `engine/GAME-ONNX/GAME-1.0.3-medium-onnx/` | [openvpi/GAME](https://github.com/openvpi/GAME) 的 release **v1.0.3** 附带 ONNX 模型 |

`GAME-1.0.3-medium-onnx/` 下应有：`encoder.onnx`、`segmenter.onnx`、`estimator.onnx`、
`dur2bd.onnx`、`bd2dur.onnx`、`config.json`。

GAME 的 ONNX 模型采用 **CC BY-NC-SA 4.0** 许可（**非商业使用**），详见上游 release 说明。
缺失模型时软件仍可启动，只是自动生成音符 / F0 相关功能不可用。

---

## 3. 目录结构

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
│   │   ├── edit.py         #     编辑操作 + 层间联动 + 标签计算
│   │   ├── undo.py         #     命令栈（命令式 / 快照式）
│   │   ├── export.py       #     导出 raw/wavs + transcriptions.csv
│   │   └── paths.py        #     资源路径单一事实源
│   ├── ui/
│   │   ├── theme.py        #     深色主题
│   │   ├── main_window.py  #     主窗口（菜单 / 层信息栏 / 自动保存）
│   │   ├── views.py        #     波形 / 频谱 / 钢琴卷帘三视图 + 交互
│   │   ├── settings_dialog.py #  统一设置面板（语言 / 频谱 / 波形 / 面板 / 配色）
│   │   ├── worker.py       #     后台推理线程（RMVPE + GAME）
│   │   └── audio.py        #     音频播放（播放当前视窗）
│   └── engine/
│       ├── onnx_util.py    #     ONNX 会话（DirectML 优先 / CPU 回退）
│       ├── f0.py           #     RMVPE F0 提取 + F0 缓存
│       └── game.py         #     GAME 音符生成（压缩时间轴映射）
├── models/rmvpe/           # RMVPE 权重（自备，见第 2 节）
├── engine/GAME-ONNX/       # GAME ONNX 模型（自备，见第 2 节）
└── assets/dictionaries/    # 内置音素词典（可直接编辑，见第 7 节）
```

软件级设置写在**软件根目录**的 `app_settings.json`（跟随软件而不跟随工程，可用环境变量
`DSL_APP_SETTINGS` 改到别处）。某些目录会由软件自动创建，不必手动准备。

---

## 4. 工程类型

软件按**目录结构**自动判定工程类型：

| 类型 | 数据目录内容 | 工程文件 | 工作数据 |
|---|---|---|---|
| **标准工程** | `wav/` + `TextGrid/`（每句同名） | `<数据目录>/project.json` | `<数据目录>/.dsl/` |
| **翻新工程** | `transcriptions.csv` + `wavs/`（或 `wav/`、根下 `*.wav`） | `<软件根>/refurbish/<名>.refurbish.json` | `<软件根>/refurbish/cache/<工程名>/` |

翻新工程用于处理**已导出**的 DiffSinger 数据集（没有 TextGrid 三层，只有 csv + 音频）：
软件**绝不往数据目录写任何东西**，工程文件与 F0 缓存都放在自己的 `refurbish/` 下。

两类工程的区别不只是入口：**翻新工程没有 words / texts 层**，词 / 短语相关编辑与
「按词显示音符标签」不适用，音符标签退回按元音核心的启发式（见第 6 节）。

---

## 5. 使用

1. **工程菜单**：`新建工程…`（Ctrl+N）/ `打开工程…`（Ctrl+O）/ `最近工程`（路径失效则置灰）。
   首次打开需填**主语言**，随后自动后台批量预处理（RMVPE F0 + GAME 音符，进度在状态栏）。
   `保存工程`（Ctrl+S，另有自动保存）。`工具`菜单可 `批量预处理` / `重跑当前句` / `批量替换音素…`。
2. **顶部层信息栏**：显示当前句、时长、音素数 / 音符数，以及选中 / 悬停音素与选中音符的
   起止、时长、音名；同时也是 **words / texts 层的编辑面板**（见下）。
3. **面板**：波形、mel 频谱、钢琴卷帘（含 F0 曲线）三者同轴对齐，可用 `QSplitter` 拖拽调整；
   `显示`菜单或 设置 →「面板」页可分别开 / 关，顺序可调。
4. **编辑**：
   - **播放**：空格 = 播放 / 停止。**右键**：在频谱 / 钢琴卷帘窗内右键**未命中音符**处 → 播放该处音素；
     右键**命中音符** → 弹出菜单（设为 / 取消 连音、无声）。
   - **音素层**：**双击**或 `Ctrl+R` 重命名、`Ctrl+D` 复制、`Delete` 删除。
     **拖动边界**自动夹在相邻音素之间（不穿越、不产生零长音素），与边界齐平的音符端点同步跟随；
     SP / AP / br / sil 等无声音素以灰色块渲染，同样可编辑
   - **words / texts 层**（在层信息栏内）：左键拖条目边界；双击改名，或先点中条目再用
     `Ctrl+R` 重命名 / `Ctrl+D` 复制 / `Delete` 删除（**只有选中条目时快捷键才作用于它**，
     在画布上一点即切回音素 / 音符语义，避免混淆）。
     复制 = 条目从中点切两半、两块同名；删除 = 删掉条目，与前一相邻条目连续时区间并入前一条目
5. **层间联动**（只向上）：拖 **phones** 边界时，words / texts 里**同时间对齐**的边界一起移动；
   拖 **words** 边界时 texts 对齐边跟随、**不动 phones**；拖 **texts** 边界不联动。
   夹取范围取各层交集，联动后三层仍严格对齐。**拖动吸附**：拖 words 边界靠近 phones 边界
   （容差 6 像素）即吸附到位，拖 texts 边界可吸 words 与 phones 边界 —— 吸附**只改落点，
   下层永不被拖走**。
6. **word 名自动更新**：该 word 下音素**增减或改名**时，名字改写为音素原名集合
   （如 `wo` → `zh/w zh/uo vf`）；音素没变则**保持源标记原样**。翻新工程无 words 层，不涉及。
7. **音符**：拖动改音高（自由）、拖两端改时间（吸附音素边界）、双击切分；
   `↑` `↓` 上下移 ±1 半音，`Shift+↑/↓` ±10 音分。休止与连音按各自规则参与时长分配。
8. **缩放 / 滚动**：滚轮 = 上下滚动、`Shift+滚轮` = 水平滚动、`Ctrl+滚轮` = 时间轴缩放
   （以光标为中心）、`Alt+滚轮` = 音高方向缩放；中键拖动 = 平移画布；`Z` = 适配全曲。
   视图菜单另有 `放大` / `缩小` / `三面板均分` / `频谱配色`。
9. **撤销 / 重做**：`Ctrl+Z` / `Ctrl+Y`（或 `Ctrl+Shift+Z`）。所有编辑（含层边界拖动、
   改名、复制、删除）都是一条可撤销命令。
10. **导出**：`导出数据集…` 弹窗填**音色名称**（必填，记进工程、下次预填），输出
    `<所选文件夹>/<音色名>/raw/wavs/*.wav` + `<所选文件夹>/<音色名>/raw/transcriptions.csv`。
    翻新工程为 `导出 CSV…`。
11. **回写 TextGrid**（标准工程，手动触发）：把编辑后的 **phones / words / texts 三层**
    覆盖写回源 TG（不再由 phones 反推 words）。
12. **设置**（`设置`菜单，软件级，跨工程共用）：语言（主语言、全局无声音素）/
    频谱（窗函数、配色、频率标度等）/ 波形（音符组下的波形）/ 面板（显隐与顺序、标签字号）/
    配色（界面配色方案，含自定义）。改动即时生效。
13. **帮助** → `关于 / 快捷键`。

---

## 6. 音符标签规则

显示在**音符块外上方**（PIT / F0 曲线挡不住）；块**内**是否再写一份由
设置 →「面板」页「在音符块内显示标签文字」控制，**默认关闭**（块内文字会被 F0 曲线压住）。

- **有 words 层时**（标准工程）：标签 = 该音符**起点所在 word** 下的**音素集合**
  （如 `w uo`；音素只显示**本体、不带语言前缀**，`en/d en/ow en/n` → `d ow n`）。
  合规 TG 下一个 word 只对应一个非连音音符；若出现第 2、3 个非连音音符则**留空**。
  音素归属按**起点**判定，因此拉长音素不会改变归属。
- **无 words 层时**（翻新工程）：启发式 = `起音辅音 + 元音核心 + 到下一音节起音之前的音素`
  （元音核心 = 与该音符重叠时长最大的那个元音；相邻音节区间首尾相接 → 同一音素只显示一次）。
- 休止一律写 `R`，连音（slur）一律写 `-`。

---

## 7. 音素词典（可直接编辑）

音素分类（元音 / 辅音 / 流音 / 全局无声音素）**不写死在代码里**，唯一事实源是
`assets/dictionaries/<语言>/` 下的 txt，每个文件一行、空格分隔：

| 文件 | 含义 |
|---|---|
| `vowels.txt` | 元音（含 `SP` / `AP` 这类按韵母处理的符号） |
| `consonants.txt` | 辅音 / 声母 |
| `liquids.txt` | 流音（多音节系统用，目前仅 en：`_r l r w y`） |

内置 `zh` / `ja` / `en` / `yue` 四套。

- **加音素**：写进对应语言的 `vowels.txt` / `consonants.txt` 即可（空格分隔）。
- **删 / 改名**：直接改 txt；不在表里的音素会被标成 `unknown`，导入时高亮提示。
- 语言前缀 `en/` / `yue/` … 按对应语言的表分类；无前缀按**工程主语言**的表分类。
- 文件不存在 = 该表为空，不报错（所以 `zh` / `ja` 没有 `liquids.txt` 也正常）。

> 这些词典是**普通数据文件**，按数据集 / 音素体系自行替换即可，不需要改代码。

---

## 8. 致谢与第三方

本软件使用了以下第三方组件（权重需自行下载，不随源码分发）：

| 组件 | 用途 | 许可 |
|---|---|---|
| [GAME](https://github.com/openvpi/GAME)（openvpi） | 音符自动生成 | 代码开源；附带的 ONNX 模型为 **CC BY-NC-SA 4.0（非商业）** |
| [RMVPE](https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI) | F0 提取 | 见上游 |
| [DiffSinger](https://github.com/openvpi/DiffSinger) / [MakeDiffSinger](https://github.com/openvpi/MakeDiffSinger) | 数据格式与流水线约定（`transcriptions.csv`、`ph_num`、TextGrid 三层等） | 见各自仓库 |
| [Qt for Python (PySide6)](https://doc.qt.io/qtforpython/) | 界面框架 | LGPLv3 |
| numpy / scipy / librosa / soundfile / sounddevice / onnxruntime | 数值、音频、推理 | BSD / ISC / MIT 等，见各包 |

**注意**：GAME 的 ONNX 权重是**非商业**许可，把它用于本工具做**商业**用途前请自行确认上游条款。

**设计参考**：[VLabeler](https://github.com/sdercolin/vlabeler)（作者 sdercolin，Apache License 2.0）——
界面布局与交互设计上参考了它的部分方案（顶部层信息栏、在标注层内直接编辑条目等）；
本软件是独立的 Python + Qt 实现，**未复制其代码**。

完整的第三方归属、上游致谢与许可全文见 **[CREDITS.md](CREDITS.md)**。

---

## 9. 许可

本仓库源码采用 **Apache License 2.0** 许可，版权归 **Aotianlinglong**（GitHub: [@Aotianlinglong](https://github.com/Aotianlinglong)）所有。

```
Copyright 2026 Aotianlinglong

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0
```

全文见 [LICENSE](LICENSE) 与 [NOTICE](NOTICE)。第三方组件的许可见 [CREDITS.md](CREDITS.md)，
其中 GAME 的 ONNX 模型为 **CC BY-NC-SA 4.0（非商业）**，独立于本仓库许可。

---

## 10. 反馈与贡献

- **Bug / 功能建议**：请到 [Issues](https://github.com/Aotianlinglong/DS-Labeler/issues) 反馈，
  尽量附上软件标题栏里的版本号、操作系统与复现步骤（能附工程片段 / 截图最好）。
- **Pull Request**：欢迎，但请先开 issue 讨论，避免大改动被搁置。
- **代码约定**：Python 3.10+、中文注释、改动请保持「离屏可跑」——软件所有界面逻辑都不能依赖
  真实显示环境（便于无头测试）。
- **不要提交**：模型权重（`models/`、`engine/GAME-ONNX/`）、`app_settings.json`、
  `refurbish/` 与个人数据集（`.gitignore` 已排除）。
