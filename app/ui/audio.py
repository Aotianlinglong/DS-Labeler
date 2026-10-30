# -*- coding: utf-8 -*-
"""
音频引擎：加载 WAV、播放当前视窗音频（播放头不跟随）。
使用 sounddevice。
"""
from __future__ import annotations

import pathlib
import time

import numpy as np
import sounddevice as sd
import soundfile as sf

TONE_SR = 44100          # 试听音的合成采样率（固定；与播放音频的 sr 无关）
TONE_GAIN = 0.45         # 峰值音量（留余量，避免叠在播放音频上时削波）
TONE_FADE_S = 0.010      # 试听音的起落渐变量（秒）：按住/松开都不会「咔」


def tone_loop(midi: float, sr: int = TONE_SR) -> np.ndarray:
    """持续发声用的**无缝循环**波形：正弦基频 + 两个低幅谐波（**零采样文件**）。

    纯正弦听感太「电子」，加一点 2/3 次谐波让它更像一件乐器；谐波幅度低，
    只影响音色、不影响音高判断。

    长度取**整数个周期**（约 0.12s）：循环接缝处相位连续，按住时不会周期性「咔」。
    这里**不做**包络——循环内的包络会变成振幅调制（听起来像抖），起落渐隐改由
    播放回调按 10ms 斜坡处理。合成一条这样的波形只要几毫秒，比打包钢琴采样
    （体积 + 变调 + 首帧 IO）更省，按下即响。
    """
    f = 440.0 * (2.0 ** ((float(midi) - 69.0) / 12.0))     # A4 = MIDI 69 = 440Hz
    cycles = max(2, int(round(0.12 * f)))                  # 目标约 0.12s 的整数周期
    n = max(8, int(round(cycles * sr / float(f))))
    t = np.arange(n, dtype=np.float64) / float(sr)
    w = (np.sin(2.0 * np.pi * f * t)
         + 0.32 * np.sin(4.0 * np.pi * f * t)
         + 0.14 * np.sin(6.0 * np.pi * f * t))
    w = w / max(1e-9, float(np.abs(w).max()))      # 谐波叠加后峰值随相位变化，先归一化
    return (w * TONE_GAIN).astype(np.float32)


_TONE_RMS = float(np.sqrt(np.mean(tone_loop(69) ** 2)))   # 归一化循环波的 RMS（响度标定）


class AudioEngine:
    """单例式音频引擎（主窗口持有一个实例）"""

    def __init__(self):
        self._data: np.ndarray | None = None      # float32 mono
        self._sr: int = 44100
        self._playing = False
        self._play_start_sample: int = 0          # 播放起始采样点（视窗起点）
        self._play_wall: float = 0.0              # 播放开始的墙钟（perf_counter）
        self._play_len: float = 0.0               # 播放段时长（秒）
        # ---- 试听音（琴键列按住发声）：一条常驻流 + 无缝循环缓冲 ----
        self._tone_stream = None                  # sd.OutputStream（懒创建）
        self._tone_buf: np.ndarray | None = None  # 当前音高的循环波形
        self._tone_key: int = -1                  # 当前循环波形对应的 MIDI
        self._tone_pos: int = 0                   # 循环读指针（采样）
        self._tone_on: bool = False               # 按住=True / 松开=False（淡出后停流）
        self._tone_gain: float = 0.0              # 当前渐变量（0..1）
        self._tone_scale = 1.0                    # 试听音量缩放（跟随当前音频的响度）

    # ---- 加载 ----------------------------------------------------------
    def load(self, wav_path: pathlib.Path) -> None:
        data, sr = sf.read(str(wav_path), dtype="float32", always_2d=True)
        if data.shape[1] > 1:
            data = data.mean(axis=1)
        else:
            data = data[:, 0]
        self._data = data
        self._sr = sr
        self.stop()
        self._tone_scale = self._loudness_scale(data, sr)
        try:
            # 预热音频设备（WASAPI 流首次打开耗时，会造成首次播放明显卡顿）
            sd.play(np.zeros(16, dtype="float32"), sr)
        except Exception:
            pass

    @property
    def loaded(self) -> bool:
        return self._data is not None

    @staticmethod
    def _loudness_scale(data: np.ndarray, sr: int) -> float:
        """试听音量按**当前音频的人声响度**缩放：既不会盖过素材，也不会小到听不见。

        语音素材里停顿/静音常占三到五成，直接拿整段 RMS 会严重低估响度，所以
        按 50ms 分帧、只取**有声帧**（高于 -60dBFS）的 RMS 中位数作为「总体响度」，
        再把试听音缩放到与它相当。
        """
        hop = max(64, int(0.05 * sr))
        n = int(len(data) // hop)
        if n < 2:
            return 1.0
        frames = np.abs(data[:n * hop]).reshape(n, hop)
        rms = np.sqrt(np.mean(frames ** 2, axis=1))
        voiced = rms[rms > 1e-3]              # -60 dBFS 以上才算「有声」
        if voiced.size == 0:
            return 1.0                        # 全静音 → 用默认音量
        level = float(np.median(voiced))
        # 目标响度：与素材人声相当；上下限保证「别太小」「别盖过素材」
        target = min(0.30, max(0.06, level))
        return min(1.6, max(0.2, target / max(1e-6, _TONE_RMS)))

    @property
    def duration(self) -> float:
        return len(self._data) / self._sr if self._data is not None else 0.0

    # ---- 播放（范围播放：只播 [t0, t1)）----------------------------------
    def play_range(self, t0: float, t1: float) -> None:
        if self._data is None:
            return
        self._silence_tone()      # 试听流先让出设备，避免和播放音频抢输出
        self.stop()
        i0 = max(0, int(t0 * self._sr))
        i1 = min(len(self._data), int(t1 * self._sr))
        if i1 <= i0:
            return
        self._play_start_sample = i0
        self._play_len = (i1 - i0) / self._sr
        self._play_wall = time.perf_counter()
        try:
            sd.play(self._data[i0:i1], self._sr)
            self._playing = True
        except Exception as e:  # 无音频设备时降级
            print(f"[audio] 播放失败: {e}")
            self._playing = False

    # ---- 试听音（琴键列按住）--------------------------------------------
    def start_tone(self, midi: float) -> None:
        """按下琴键：开始持续发声（松开前一直响）。

        用一条循环播放的 OutputStream 而不是 `sd.play` 一次性播一段：按住多久
        就响多久、没有重复触发的接缝。正在播放音频时先停掉（两条流会抢设备）。
        """
        if self._playing:
            self.stop()
        key = int(round(midi))
        if self._tone_buf is None or self._tone_key != key:
            self._tone_buf = tone_loop(key)
            self._tone_key = key
        self._tone_pos = 0
        self._tone_on = True
        try:
            self._ensure_tone_stream()
            if not self._tone_stream.active:      # 被 _silence_tone 停过（播放音频时让出设备）→ 重启
                self._tone_stream.start()
        except Exception as e:  # 无音频设备时降级
            print(f"[audio] 试听失败: {e}")

    def stop_tone(self) -> None:
        """松开琴键：淡出到静音（流保持运行，下次按键立即发声，不用重启流）。"""
        self._tone_on = False

    def _silence_tone(self) -> None:
        """立即掐掉试听音（播放音频前调用，避免两条流同时占设备）。"""
        self._tone_on = False
        self._tone_gain = 0.0
        if self._tone_stream is not None and self._tone_stream.active:
            try:
                self._tone_stream.stop()
            except Exception:
                pass

    def _ensure_tone_stream(self) -> None:
        if self._tone_stream is None:
            self._tone_stream = sd.OutputStream(
                samplerate=TONE_SR, channels=1, dtype="float32",
                blocksize=512, callback=self._tone_cb)

    def _tone_cb(self, outdata, frames, time_info, status) -> None:
        """循环送音 + 10ms 起落斜坡。

        松开后只把增益淡到 0、继续输出静音，**不停流**——PortAudio 里因回调
        自行结束的流无法再次 `start()`（实测静默失败，表现为「按一次之后就不响」），
        常驻静音流才能保证下一次按键立即出声；需要让出设备时由 `_silence_tone` 显式停。
        """
        buf = self._tone_buf
        if buf is None or (not self._tone_on and self._tone_gain <= 0.0):
            outdata.fill(0.0)
            return
        n = len(buf)
        idx = (self._tone_pos + np.arange(frames, dtype=np.int64)) % n
        x = buf[idx]
        step = 1.0 / max(1.0, TONE_FADE_S * TONE_SR)
        if self._tone_on:
            g = np.minimum(self._tone_gain + step * np.arange(1, frames + 1), 1.0)
        else:
            g = np.maximum(self._tone_gain - step * np.arange(1, frames + 1), 0.0)
        self._tone_gain = float(g[-1])
        self._tone_pos = int((self._tone_pos + frames) % n)
        y = x * g * self._tone_scale      # 音量跟随当前音频的人声响度
        if outdata.ndim > 1:
            outdata[:, 0] = y
        else:
            outdata[:] = y

    def stop(self) -> None:
        self._silence_tone()
        if self._playing:
            try:
                sd.stop()
            except Exception:
                pass
        self._playing = False

    def close(self) -> None:
        """释放试听流（程序退出时调用；静音流可能仍在运行）。"""
        self._tone_on = False
        self._tone_gain = 0.0
        if self._tone_stream is not None:
            try:
                self._tone_stream.stop()
                self._tone_stream.close()
            except Exception:
                pass
            self._tone_stream = None

    def toggle(self, t0: float, t1: float) -> bool:
        """播放/暂停。返回当前是否在播放。"""
        if self._playing:
            self.stop()
        else:
            self.play_range(t0, t1)
        return self._playing

    @property
    def playing(self) -> bool:
        return self._playing

    def current_pos(self) -> float:
        """当前播放位置（秒）。

        用本地墙钟推算（sd.play 是 fire-and-forget，get_stream().time
        在隐式流下不可靠，会导致播放头永远不动）。播放到段尾自动复位，
        并返回段尾时刻让播放头停在末尾。
        """
        if not self._playing:
            return -1.0
        el = time.perf_counter() - self._play_wall
        if el >= self._play_len:
            self._playing = False
            return self._play_start_sample / self._sr + self._play_len
        return self._play_start_sample / self._sr + el
