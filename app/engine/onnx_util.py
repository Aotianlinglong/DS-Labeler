# -*- coding: utf-8 -*-
"""
ONNX 运行环境工具。

约定：onnxruntime 必须已导入（由 app/main.py 的入口守卫保证）。
引擎内部只在这里创建 Session，DirectML(GPU) 优先，失败自动回退 CPU。
"""
from __future__ import annotations

import pathlib

import numpy as np

DML_PROVIDER = "DmlExecutionProvider"
CPU_PROVIDER = "CPUExecutionProvider"

# 记录实际使用的 provider，便于日志/状态栏显示
_last_provider: str = ""


def available_providers() -> list[str]:
    try:
        import onnxruntime as ort
        return list(ort.get_available_providers())
    except Exception:
        return [CPU_PROVIDER]


def last_provider() -> str:
    return _last_provider


def make_session(model_path: str | pathlib.Path, prefer_gpu: bool = True,
                 intra_threads: int = 4):
    """创建 ONNX 会话：GPU(DML) 优先，回退 CPU。"""
    global _last_provider
    import onnxruntime as ort

    so = ort.SessionOptions()
    so.intra_op_num_threads = max(1, intra_threads)
    so.log_severity_level = 3

    providers: list = []
    if prefer_gpu and DML_PROVIDER in available_providers():
        providers.append((DML_PROVIDER, {"device_id": 0}))
    providers.append(CPU_PROVIDER)

    sess = ort.InferenceSession(str(model_path), sess_options=so, providers=providers)
    try:
        _last_provider = sess.get_providers()[0]
    except Exception:
        _last_provider = CPU_PROVIDER
    return sess


def resample(x: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    """librosa 重采样，返回 float32。"""
    x = np.asarray(x, dtype=np.float32)
    if orig_sr == target_sr:
        return x
    import librosa
    return librosa.resample(x, orig_sr=orig_sr, target_sr=target_sr).astype(np.float32)
