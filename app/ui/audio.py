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


class AudioEngine:
    """单例式音频引擎（主窗口持有一个实例）"""

    def __init__(self):
        self._data: np.ndarray | None = None      # float32 mono
        self._sr: int = 44100
        self._playing = False
        self._play_start_sample: int = 0          # 播放起始采样点（视窗起点）
        self._play_wall: float = 0.0              # 播放开始的墙钟（perf_counter）
        self._play_len: float = 0.0               # 播放段时长（秒）

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
        try:
            # 预热音频设备（WASAPI 流首次打开耗时，会造成首次播放明显卡顿）
            sd.play(np.zeros(16, dtype="float32"), sr)
        except Exception:
            pass

    @property
    def loaded(self) -> bool:
        return self._data is not None

    @property
    def duration(self) -> float:
        return len(self._data) / self._sr if self._data is not None else 0.0

    # ---- 播放（范围播放：只播 [t0, t1)）----------------------------------
    def play_range(self, t0: float, t1: float) -> None:
        if self._data is None:
            return
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

    def stop(self) -> None:
        if self._playing:
            try:
                sd.stop()
            except Exception:
                pass
        self._playing = False

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
