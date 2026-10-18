# CREDITS / 致谢与第三方归属

**Dataset Labeler（DS-Labeler）**
Copyright 2026 Aotianlinglong · <https://github.com/Aotianlinglong/DS-Labeler>
许可：Apache License 2.0（源码，见 [LICENSE](LICENSE) / [NOTICE](NOTICE)）

本软件站在一批开源项目与研究工作的肩膀上，特此致谢。下列第三方组件**按需自行获取**，
本仓库不打包它们的权重或代码。

---

## 1. 模型与推理

### GAME —— 音符自动生成
- 项目：<https://github.com/openvpi/GAME>（openvpi）
- 本软件使用其 release **v1.0.3** 附带的 ONNX 模型（`GAME-1.0.3-medium-onnx`，medium 规格）
- **模型许可：CC BY-NC-SA 4.0（署名 - 非商业性使用 - 相同方式共享）**
  —— 用于商业用途前请自行确认上游条款；该许可独立于本仓库的 Apache-2.0
- 上游数据标注致谢（转引自其 release 说明）：[挚彬Club](https://zhibin.club/)、
  [YQ之神](https://space.bilibili.com/102844209)、小狼、[幽寂](https://space.bilibili.com/478860)

### RMVPE —— F0（基频）提取
- 本软件使用 `rmvpe.onnx`，经由
  [RVC-Project / Retrieval-based-Voice-Conversion-WebUI](https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI)
  分发的 ONNX 导出获得
- 权重下载：<https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/rmvpe.onnx>
- 许可：见上游项目与模型分发页

### R3MOE —— 口型（开口度）曲线提取
- 项目：<https://github.com/KakaruHayate/R3MOE>（KakaruHayate）
- **许可：MPL-2.0**（Mozilla Public License 2.0）
- 说明：本软件使用其导出的 ONNX 模型（默认文件名 `opec.onnx`）计算「口型面板」的开口度曲线。
  上游只发布训练工程与导出脚本，**ONNX 需自行导出**；本仓库**不打包其权重，也未包含其源代码**，
  只在运行时通过 ONNX Runtime 调用。输入为 44.1 kHz 单声道波形，输出为 50 帧 / 秒的归一化开口度。
- 该权重**可缺**：缺失时口型面板无内容，其余功能不受影响。

### ONNX Runtime
- 项目：<https://github.com/microsoft/onnxruntime>（Microsoft），许可 MIT
- Windows 下使用 `onnxruntime-directml` 走 DirectML（DX12）GPU 推理

---

## 2. 生态与格式约定

### DiffSinger / MakeDiffSinger
- 项目：<https://github.com/openvpi/DiffSinger>、<https://github.com/openvpi/MakeDiffSinger>
- 本软件遵循其数据集约定：`raw/wavs + transcriptions.csv`、`ph_num`（音素到词的归并）、
  TextGrid 的 `texts` / `words` / `phones` 三层结构
- `ph_num` 的推导口径有意与上游 `add_ph_num.py`（以及多音节系统用的 advanced 版本）保持一致
- 内置音素词典的**元音 / 辅音 / 流音分类体系**参考 DiffSinger 生态的音素约定；
  具体分类表由本仓库维护者整理，作为普通数据文件提供，可直接替换

### Praat / TextGrid
- TextGrid 是 Praat 的标注文件格式（<https://www.fon.hum.uva.nl/praat/>）；
  本软件自带读写实现，不包含 Praat 代码

---

## 3. 运行依赖

| 包 | 用途 | 许可 |
|---|---|---|
| [Qt for Python (PySide6)](https://doc.qt.io/qtforpython/) | 界面框架（Qt6） | LGPLv3（Qt 另有商业许可可选） |
| [numpy](https://numpy.org/) | 数值计算 | BSD-3-Clause |
| [scipy](https://scipy.org/) | 信号处理 | BSD-3-Clause |
| [librosa](https://librosa.org/) | STFT / mel 频谱 | ISC |
| [soundfile](https://github.com/bastibe/python-soundfile) | 音频读写（libsndfile 绑定） | BSD-3-Clause |
| [sounddevice](https://github.com/spatialaudio/python-sounddevice) | 音频播放（PortAudio 绑定） | MIT |
| [onnxruntime](https://github.com/microsoft/onnxruntime) | 模型推理 | MIT |

---

## 4. 设计参考

本仓库的源码为独立实现，下列项目**未被复制代码**，但其功能设计或交互方案给了本软件启发，
特此致谢。

### VLabeler —— 界面与交互设计参考
- 项目：<https://github.com/sdercolin/vlabeler>（作者 sdercolin，主页 <https://vlabeler.com>）
- 许可：Apache License 2.0
- 说明：本软件在**界面布局与交互设计**上参考了 VLabeler 的部分方案，例如顶部层信息栏、
  在标注层内直接编辑条目（拖边界 / 双击改名 / 快捷键增删）等。
  VLabeler 是 Kotlin 实现，本仓库是**独立的 Python + Qt 实现**，未复制其源码。

---

## 5. 项目作者与反馈

- 作者 / 维护者：**Aotianlinglong**（<https://github.com/Aotianlinglong>）
- 问题反馈与功能建议：<https://github.com/Aotianlinglong/DS-Labeler/issues>
- 欢迎 Pull Request，建议先开 issue 讨论

若您是上述项目 / 数据的作者，认为本仓库的署名或引用方式有误或需调整，请开 issue 指出，
我们会尽快修正。
