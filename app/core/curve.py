# -*- coding: utf-8 -*-
"""
R3MOE 口型（开口度）曲线提取（ONNX）。

上游：<https://github.com/KakaruHayate/R3MOE>（作者 KakaruHayate，许可 MPL-2.0）。
本模块是**独立实现**（只调 onnxruntime，未复制上游任何代码 / 未使用 torch）。
权重（onnx）由用户自备，放 `models/r3moe/opec.onnx`（见 `core.paths.MOUTH_MODEL`、
README「模型放置」节）；本仓库不打包权重。

模型事实（实测 ONNX 签名 + 上游 `export_onnx.py`）：
  · 输入 `waveform` [batch, time] float32，**必须 44.1kHz 单声道** —— 上游 wrapper 把
    `input_sr` 硬编码成 44100，内部再重采样到模型原生的 16kHz；故喂之前若源采样率不是
    44100 要先重采样。
  · 输出 `curve` [batch, frames] float32，值 = `sigmoid*1.15 − 0.15 ∈ (−0.15, 1.0)`，
    即**归一化开口度**（1 = 最大张口；0 附近 = 闭嘴）。
  · 帧率 = 16000 / 320 = **50 fps**，center=True → 第 i 帧 ↔ t = i/50 s（**无时间偏移**）。

推理只需 onnxruntime，不需要 torch / torchaudio / librosa（重采样用 engine.onnx_util 的
librosa 实现，仅在源采样率 != 44100 时才会用到）。
缓存：`<工作目录>/curve/<句名>.npz`（curve / fps）。
"""
from __future__ import annotations

import pathlib

import numpy as np

from engine.onnx_util import make_session, resample

SR_IN = 44100          # 模型要求的输入采样率（上游 wrapper 硬编码）
FPS = 50.0             # 帧率 = 16000 / 320


class MouthCurveExtractor:
    """R3MOE ONNX 包装（懒加载，进程内复用）。

    走 **CPU**：模型是小型 CNN + 双层 BiLSTM（约 40MB），CPU 足够快，且省去 DML 的
    算子切分与显存拷贝（与 RMVPE 相同口径，见 engine.f0）。
    """

    def __init__(self, model_path: str | pathlib.Path, prefer_gpu: bool = False):
        self.model_path = str(model_path)
        self.prefer_gpu = prefer_gpu
        self._session = None

    def available(self) -> bool:
        """权重是否就位（缺权重时软件仍可用，只是口型面板无内容）。"""
        return pathlib.Path(self.model_path).exists()

    def _ensure(self):
        if self._session is None:
            self._session = make_session(self.model_path, self.prefer_gpu)

    def extract(self, samples: np.ndarray, sr: int) -> np.ndarray:
        """任意采样率单声道 float32 → 口型曲线（float32，帧率 50fps）。"""
        self._ensure()
        x = resample(np.asarray(samples, dtype=np.float32).reshape(-1), int(sr), SR_IN)
        waveform = np.ascontiguousarray(x[None, :], dtype=np.float32)
        out = self._session.run(["curve"], {"waveform": waveform})[0]
        return np.asarray(out[0], dtype=np.float32)


# ---- 缓存读写 -------------------------------------------------------------
def save_curve(path: str | pathlib.Path, curve: np.ndarray, fps: float = FPS) -> None:
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, curve=np.asarray(curve, dtype=np.float32),
                        fps=np.float32(fps))


def load_curve(path: str | pathlib.Path) -> tuple[np.ndarray, float]:
    with np.load(str(path)) as z:
        return z["curve"].astype(np.float32), float(z["fps"])
