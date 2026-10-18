# -*- coding: utf-8 -*-
"""
频谱计算（纯 numpy/librosa，无 Qt 依赖，便于离线验证）。

绘制方法（FFT / 窗函数 / 频率轴等的默认参数）：
  - FFT 间隔 hop = 110；窗长按采样率自适应（见 pick_n_fft，不暴露给设置面板）；
    窗函数默认 Blackman-Harris（副瓣 -92dB），可在「频谱设置」里改
  - 频率轴**可选**（mel / 线性 Hz / 对数 Hz；见 SCALES）——并不限于 mel，
    故这里做成可配置；默认仍是 mel（感知均匀）
  - 频率分辨率 10（每 10 mel 一行 → 0~20kHz 约 382 行；换频率轴时行数不变、只改分布）
  - 最大显示频率 20000 Hz
  - 强度范围固定 [-20, 55] dB（full-scale 归一）
  - 配色默认 Viridis
  - 点数/像素 2：整段一次算完缓存，视图缩放时均值/线性插值重采样等效

以上参数除窗长外都可由「设置 → 频谱设置…」调整（存入工程 ds_labeler_settings.json）。

整段一次算完并缓存：`SpecData` 保存整句的频谱（n_rows × n_frames），
视图只做「按时间切片 + 重采样」，平移/缩放不再重算。
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# ---- 频谱图默认参数 ----
N_FFT = 0               # 0 = 按采样率自适应（见 pick_n_fft）：44.1kHz → 4096
HOP = 110               # FFT 间隔 110 → 401 列/秒
WINDOW = "blackmanharris"   # 默认窗（副瓣 -92dB：谐波锐利、谷底干净）
MAX_FREQ = 20000.0      # 最大显示频率
MEL_RES = 10            # 频率分辨率（mel）：每 10 mel 一显示行
DB_MIN = -20.0          # 最小显示强度
DB_MAX = 55.0           # 最大显示强度

# ---- 频率轴（行 → 频率 的映射）可选 ----
# 三者的**行数完全相同**（都由 mel 分辨率决定），差别只在频率怎么分布到这些行上：
#   mel    = 感知均匀（低频铺得开、高频压缩）
#   linear = 等 Hz 间隔（每行固定 Hz 数），看谐波等间距最直观
#   log    = 对数 Hz（每行固定频率比），低频细节与高频兼顾
SCALES = ("mel", "linear", "log")
SCALE_LABELS = {"mel": "mel（感知均匀）", "linear": "线性 Hz（等间隔）",
                "log": "对数 Hz（等比例）"}
LOG_FMIN = 20.0         # 对数轴的下限频率（避免 log0）

# 可选的窗函数（窗长固定按采样率自适应，不暴露）
WINDOWS = ("blackmanharris", "hann", "hamming", "blackman", "bartlett", "flattop")
WINDOW_NAMES = {
    "blackmanharris": "Blackman-Harris（默认）",
    "hann": "Hann（Hanning）",
    "hamming": "Hamming",
    "blackman": "Blackman",
    "bartlett": "Bartlett（三角）",
    "flattop": "Flat-top（幅值最准）",
}

# ---- 调色板（256 级 RGB；默认 viridis）----
_CMAP_STOPS = {
    "viridis": [(12, 5, 51), (50, 20, 90), (70, 70, 130), (50, 130, 120),
                (90, 190, 80), (190, 220, 60), (250, 250, 120)],
    "soft":    [(0, 0, 0), (8, 20, 70), (20, 70, 140), (60, 150, 200),
                (140, 220, 240), (240, 252, 255)],
    "rust":    [(0, 0, 0), (60, 15, 5), (140, 40, 10), (210, 110, 30),
                (250, 200, 90), (255, 250, 220)],
    "magma":   [(0, 0, 4), (40, 12, 60), (120, 28, 109), (201, 65, 109),
                (241, 130, 96), (252, 200, 120), (252, 253, 191)],
    "gray":    [(0, 0, 0), (170, 170, 170), (255, 255, 255)],
}
CMAPS = tuple(_CMAP_STOPS.keys())
DEFAULT_CMAP = "viridis"
CMAP_LABELS = {"soft": "柔和 soft", "rust": "铁锈 rust",
               "magma": "暖色 magma", "viridis": "冷色 viridis", "gray": "灰度 gray"}

_cmap_cache: dict[str, np.ndarray] = {}


def make_cmap(name: str) -> np.ndarray:
    """生成 256×3 uint8 调色板。"""
    if name in _cmap_cache:
        return _cmap_cache[name]
    stops = _CMAP_STOPS.get(name, _CMAP_STOPS[DEFAULT_CMAP])
    xs = np.linspace(0, 1, len(stops))
    t = np.linspace(0, 1, 256)
    pal = np.zeros((256, 3), dtype=np.uint8)
    for c in range(3):
        pal[:, c] = np.interp(t, xs, [s[c] for s in stops]).astype(np.uint8)
    _cmap_cache[name] = pal
    return pal


def _hz_to_mel(f: np.ndarray | float) -> np.ndarray | float:
    """HTK mel：2595·log10(1 + f/700)。"""
    return 2595.0 * np.log10(1.0 + np.asarray(f, dtype=np.float64) / 700.0)


def _mel_to_hz(m: np.ndarray | float) -> np.ndarray | float:
    return 700.0 * (10.0 ** (np.asarray(m, dtype=np.float64) / 2595.0) - 1.0)


def pick_n_fft(sr: int) -> int:
    """按采样率自适应选 FFT 长度，使 Blackman-Harris 主瓣（≈8 bin）不宽于 ~110 Hz。

    这是「谐波糊成一块」的决定性因素：谐波间距 = F0（人声常见 80~400 Hz）。
    若主瓣宽于谐波间距，相邻谐波在 STFT 阶段就已被互相吞掉，之后再怎么渲染都糊。
    44.1kHz → 4096（主瓣 86 Hz）；22.05kHz → 2048；96kHz → 8192。
    """
    for n in (1024, 2048, 4096, 8192):
        if 8.0 * sr / n <= 110.0:
            return n
    return 8192


@dataclass
class SpecData:
    """整段频谱：data = uint8 [n_rows, n_frames]，行 0 = 低频。"""
    data: np.ndarray          # (n_rows, n_frames) uint8
    fps: float                # 每秒帧数 = sr / hop
    sr: int
    hop: int = HOP
    n_rows: int = 0
    fmax: float = MAX_FREQ
    n_fft: int = 0            # 实际使用的 FFT 长度（诊断用）
    window: str = WINDOW      # 实际使用的窗函数（诊断用）
    scale: str = "mel"        # 实际使用的频率轴（诊断用）

    @property
    def n_frames(self) -> int:
        return int(self.data.shape[1])

    @property
    def duration(self) -> float:
        return self.n_frames / self.fps if self.fps > 0 else 0.0

    def slice_px(self, t0: float, t1: float) -> tuple[np.ndarray, float, float]:
        """取 [t0,t1) 的列，返回 (block[n_rows, k], 首列时间, 末列时间)。"""
        f0 = int(np.floor(max(0.0, t0) * self.fps))
        f1 = int(np.ceil(max(0.0, t1) * self.fps)) + 1
        f0 = max(0, min(f0, self.n_frames))
        f1 = max(f0 + 1, min(f1, self.n_frames))
        return self.data[:, f0:f1], f0 / self.fps, (f1 - 1) / self.fps


def spec_from_audio(y: np.ndarray, sr: int, *, n_fft: int = N_FFT, hop: int = HOP,
                    window: str = WINDOW, max_freq: float = MAX_FREQ,
                    mel_res: float = MEL_RES, scale: str = "mel",
                    db_min: float = DB_MIN, db_max: float = DB_MAX
                    ) -> SpecData:
    """整段音频 → SpecData。

    算法要点：
      1. 幅度用 16-bit PCM 整数标度（float ±1.0 × 32768，不除 32768）
      2. 两端补 n_fft/2 零（帧中心对齐 k·hop；librosa center=True 等价）
      3. 对称窗（GeneralCosine sym=true，非周期窗）
      4. dB = 20·log10(|X| / windowSize)，不除 32768、不乘 2、非功率谱
      5. 频率截断：maxYLength = int((N/2+1)·maxFreq·2/sr)，
         displayIndex = freq·(maxYLength-1)/maxFreq（按上述分母映射）
      6. 频率轴（行 → 频率）：行数固定由 mel_res 决定，频率分布按 scale 三选一：
         mel（HTK，每行 mel_res）、linear（0~fmax 等 Hz）、log（等比）
      7. 取行值 = floor/ceil 两 bin 线性插值，越界取 0
      8. 强度线性映射 [db_min, db_max] → [0,1]，先 clamp 无 gamma
    """
    import librosa
    from scipy.signal import get_window

    if not n_fft:
        n_fft = pick_n_fft(sr)                                        # 自适应窗长（见 pick_n_fft）
    y = np.asarray(y, dtype=np.float64).reshape(-1) * 32768.0        # PCM 整数标度
    fps = sr / hop
    if y.size == 0:
        return SpecData(np.zeros((1, 1), dtype=np.uint8), fps=fps, sr=sr,
                        hop=hop, n_rows=1, fmax=max_freq)
    wname = window if window in WINDOWS else WINDOW
    win = get_window(wname, n_fft, fftbins=False)                    # 对称窗（sym=true）
    spec = librosa.stft(y=y, n_fft=n_fft, hop_length=hop, window=win, center=True)
    mag = np.abs(spec)                                                # (N/2+1, n_frames)
    db = 20.0 * np.log10(np.maximum(mag, 1e-12) / n_fft)              # 20log10(|X|/windowSize)

    # sr>44100 的音频常规做法是重采样到 44100 再算；这里直接把显示上限 clamp 到 Nyquist
    eff_freq = float(min(max_freq, sr / 2.0))
    # 频率截断（把截断后的数组当作铺满 0~Nyquist 来映射 bin）
    n_bins_full = mag.shape[0]
    max_y_len = max(2, min(n_bins_full, int(n_bins_full * eff_freq * 2.0 / sr)))
    max_mel = int(_hz_to_mel(eff_freq))
    mels = list(range(0, max(1, max_mel // int(mel_res)) * int(mel_res) + 1,
                      int(mel_res))) + [max_mel]
    # 频率轴：行数固定（由 mel_res 决定竖直密度），频率分布按 scale 换
    sc = scale if scale in SCALES else "mel"
    n_rows = len(mels)
    if sc == "linear":
        freqs = np.linspace(0.0, eff_freq, n_rows)
    elif sc == "log":
        freqs = np.geomspace(min(LOG_FMIN, eff_freq), eff_freq, n_rows)
    else:
        freqs = _mel_to_hz(np.asarray(mels, dtype=np.float64))
    bin_pos = freqs * (max_y_len - 1) / eff_freq                      # displayIndex
    nb = max_y_len - 1
    i0 = np.floor(bin_pos).astype(np.int64)
    i1 = i0 + 1
    frac = (bin_pos - i0).astype(np.float32)[:, None]
    valid0 = (i0 >= 0) & (i0 <= nb)
    valid1 = (i1 >= 0) & (i1 <= nb)
    i0c = np.clip(i0, 0, max(nb, 0))
    i1c = np.clip(i1, 0, max(nb, 0))
    v0 = np.where(valid0[:, None], db[i0c], -db_max)                  # 越界取 0（黑）
    v1 = np.where(valid1[:, None], db[i1c], -db_max)
    dbm = v0 * (1.0 - frac) + v1 * frac                               # 频率方向线性插值

    norm = np.clip((dbm - db_min) / (db_max - db_min), 0.0, 1.0)
    data = (norm * 255.0 + 0.5).astype(np.uint8)
    return SpecData(data, fps=fps, sr=sr, hop=hop, n_rows=int(data.shape[0]),
                    fmax=float(max_freq), n_fft=int(n_fft), window=wname, scale=sc)
