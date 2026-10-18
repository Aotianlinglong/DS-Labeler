# -*- coding: utf-8 -*-
"""
RMVPE F0 提取（ONNX，DirectML GPU 优先）。

- 输入：任意采样率 mono float32；内部重采样到 16kHz
- 输出：f0（Hz，未发声帧为 0）+ uv（bool，**True = 未发声**，与 RMVPE 参考一致），
  帧长固定 0.01s；绘制/判声请用 f0 > 0
- 缓存：`.dsl/f0/<句名>.npz`（f0/uv/hop），打开工程秒载不重跑
"""
from __future__ import annotations

import pathlib

import numpy as np

from engine.onnx_util import make_session, resample

RMVPE_SR = 16000          # RMVPE 要求 16k
RMVPE_HOP = 0.01          # 帧长 10ms
RMVPE_THRESHOLD = 0.03    # 与 rmvpe 参考实现一致


class F0Extractor:
    """RMVPE ONNX 包装（懒加载，进程内复用）。"""

    def __init__(self, model_path: str | pathlib.Path, prefer_gpu: bool = True):
        self.model_path = str(model_path)
        self.prefer_gpu = prefer_gpu
        self._session = None

    def _ensure(self):
        if self._session is None:
            self._session = make_session(self.model_path, self.prefer_gpu)

    def extract(self, samples: np.ndarray, sr: int) -> tuple[np.ndarray, np.ndarray]:
        """返回 (f0[Hz], uv[bool])，帧长 0.01s。"""
        self._ensure()
        x = resample(samples, sr, RMVPE_SR)
        waveform = x[None, :].astype(np.float32)
        threshold = np.array(RMVPE_THRESHOLD, dtype=np.float32)
        f0, uv = self._session.run(
            ["f0", "uv"], {"waveform": waveform, "threshold": threshold})
        return f0[0].astype(np.float32), uv[0].astype(bool)


# ---- 缓存读写 -------------------------------------------------------------
def save_f0(path: str | pathlib.Path, f0: np.ndarray, uv: np.ndarray,
            hop: float = RMVPE_HOP) -> None:
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, f0=f0.astype(np.float32), uv=uv.astype(bool),
                        hop=np.float32(hop))


def load_f0(path: str | pathlib.Path) -> tuple[np.ndarray, np.ndarray, float]:
    with np.load(str(path)) as z:
        return (z["f0"].astype(np.float32), z["uv"].astype(bool), float(z["hop"]))
