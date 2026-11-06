# -*- coding: utf-8 -*-
"""传统单音高判定：切分产生的新音符用它把 MIDI 对齐到实际人声。

修音软件（Auto-Tune / Melodyne 一类）的做法是：逐帧估计基频 F0，再由帧级结果
定音符音高。已加载的 F0 曲线（如 RMVPE 缓存）可直接查表取值（`estimate_midi_from_f0`），
不读音频、更省也更稳；没有曲线时退回现场跑 YIN（`estimate_note_midi`）。
YIN 用 de Cheveigné & Kawahara (2002) 的差分函数 + 累积均值归一化 + 绝对阈值 +
抛物线插值；对歌声比纯自相关更少八度错误、精度更高。

音符音高取**有声帧 MIDI 的中位数**：切分点附近常有气声、瞬态或跑调帧，
中位数不会被个别帧带偏。判定不可靠（近乎静音 / 区间过短 / 无周期）时返回 None，
调用方保持原音高即可。
"""
from __future__ import annotations

import numpy as np

F0_MIN = 55.0          # 人声下限（约 A1），留出低音余量
F0_MAX = 1000.0        # 上限覆盖假声/哨音
YIN_THRESHOLD = 0.15   # 首个低于该值的谷即视为周期（YIN 论文推荐 0.1~0.15）
VOICED_MAX = 0.35      # 谷值仍高于此 → 判为无声/噪声帧
MIN_SEG_SEC = 0.03     # 短于 30ms 的区间不判（不足两个基音周期）
WIN_MAX = 2048         # 帧长上限（44.1kHz 约 46ms）


def _yin_f0(frame, sr: float):
    """单帧 YIN → 基频（Hz）；非周期或能量过低返回 None。"""
    x = np.asarray(frame, dtype=np.float64)
    if x.size < 4:
        return None
    x = x - x.mean()
    n = x.size
    tau_min = max(2, int(sr / F0_MAX))
    tau_max = min(n - 2, int(sr / F0_MIN))
    if tau_max <= tau_min + 1:
        return None
    if float(np.sqrt(np.mean(x * x))) < 1e-5:
        return None
    # 线性自相关（FFT 加速用零填充避免循环卷积）：
    #   d(tau) = sum_i (x[i] - x[i+tau])^2 ≈ r(0) + r_tau(0) - 2 r(tau)
    spec = np.fft.rfft(x, 2 * n)
    ac = np.fft.irfft(spec * np.conj(spec), 2 * n)[:n]
    if ac[0] <= 1e-12:
        return None
    d = np.maximum(ac[0] + ac[0] - 2.0 * ac, 0.0)
    lag = np.arange(1, tau_max + 1)
    cum = np.cumsum(d[1:tau_max + 1])
    cmnd = np.ones(tau_max + 1)
    cmnd[1:tau_max + 1] = d[1:tau_max + 1] * lag / np.maximum(cum, 1e-12)
    seg = cmnd[tau_min:tau_max]
    below = np.nonzero(seg < YIN_THRESHOLD)[0]
    if below.size:
        t = tau_min + int(below[0])
        while t + 1 <= tau_max and cmnd[t + 1] < cmnd[t]:
            t += 1                       # 走到该谷的局部最低点
    else:
        t = tau_min + int(np.argmin(seg))
    if cmnd[t] > VOICED_MAX:
        return None
    # 抛物线插值：把周期估到亚采样精度（音分级别）
    if 1 <= t < tau_max:
        a, b, c = cmnd[t - 1], cmnd[t], cmnd[t + 1]
        den = a - 2.0 * b + c
        if abs(den) > 1e-12:
            t = t + 0.5 * (a - c) / den
    return float(sr / t) if t > 0 else None


def estimate_midi_from_f0(f0, hop: float, t0: float, t1: float):
    """直接用已算好的 F0 曲线（如 RMVPE 缓存）取音符音高，不读音频。

    `f0` 是逐帧基频（0/NaN = 无声帧），`hop` 是帧间隔（秒）。
    取 [t0, t1) 内有声帧的 MIDI 中位数；区间内没有有效帧 → None。
    切分瞬间只需查表，比现场跑 YIN 更省（结果也更稳，F0 曲线已做过声学建模）。
    """
    if f0 is None or not hop or hop <= 0 or t1 <= t0:
        return None
    arr = np.asarray(f0, dtype=np.float64).reshape(-1)
    if arr.size == 0:
        return None
    i0 = max(0, int(round(float(t0) / hop)))
    i1 = min(arr.size, int(round(float(t1) / hop)))
    if i1 <= i0:
        return None
    vals = arr[i0:i1]
    vals = vals[np.isfinite(vals) & (vals >= F0_MIN) & (vals <= F0_MAX)]
    if vals.size == 0:
        return None
    midis = 69.0 + 12.0 * np.log2(vals / 440.0)
    return round(float(np.median(midis)), 2)


def estimate_note_midi(samples, sr: float, t0: float, t1: float):
    """[t0, t1) 区间的传统音高判定 → MIDI（保留到 1 音分）；不可靠时 None。

    逐帧（帧长 ≤46ms、步进 1/4 帧）跑 YIN，把有帧的 MIDI 取中位数。
    只依赖 numpy，与模型推理无关，可在双击切分的瞬间同步计算。
    """
    if samples is None or not sr or sr <= 0 or t1 <= t0:
        return None
    total = len(samples)
    i0 = int(max(0.0, float(t0)) * sr)
    i1 = int(min(float(total) / float(sr), float(t1)) * sr)
    seg = np.asarray(samples[i0:i1], dtype=np.float64)
    if seg.size < int(MIN_SEG_SEC * sr):
        return None
    win = min(WIN_MAX, seg.size)
    hop = max(64, win // 4)
    pitches = []
    for start in range(0, seg.size - win + 1, hop):
        f0 = _yin_f0(seg[start:start + win], sr)
        if f0 is not None:
            pitches.append(69.0 + 12.0 * np.log2(f0 / 440.0))
    if not pitches:
        return None
    return round(float(np.median(pitches)), 2)
