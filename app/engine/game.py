# -*- coding: utf-8 -*-
"""
GAME ONNX 音符生成（词级对齐）。

实现 GAME `infer.py align` 的推理链路，用 ONNX Runtime 驱动（无 PyTorch）：
  1) 由音素序列 + ph_num 得到「词」→ known_boundaries（GAME 的已知词边界）
  2) encoder.onnx  波形 → 隐特征 x_seg / x_est / maskT
  3) segmenter.onnx + D3PM 采样循环 → 音符边界 boundaries
  4) estimator.onnx  → 每个音符的时值 durations / 音高 scores / 发声 presence
  5) align_notes_to_words 把音符吸附到词边界 → note_seq / note_dur / note_slur

常量与参考实现一致（engine/GAME）。参数内置默认、不暴露。
"""
from __future__ import annotations

import json
import pathlib

import numpy as np

from core.notes import Note
from engine.onnx_util import make_session, resample

GAME_SR = 44100
GAME_TIMESTEP = 0.01

# 推理参数（内置默认；与 GAME infer.py 默认值一致）
D3PM_T0 = 0.0
D3PM_STEPS = 8
SEG_THRESHOLD = 0.2       # boundary_decoding_threshold
SEG_RADIUS_SEC = 0.02     # boundary_decoding_radius（秒）→ 帧 = round(0.02/0.01)
EST_THRESHOLD = 0.2       # note_presence_threshold
UV_WORD_COND = "lead"     # uv 词判定：只看词首音素
UV_NOTE_COND = "predict"  # 音符 v/uv 采用模型原判

# estimator 解码常量（来自 GAME 训练配置 inference 段）
MIDI_MIN = 0.0
MIDI_MAX = 128.0
MIDI_STD = 0.5


def d3pm_ts() -> list[float]:
    step = (1.0 - D3PM_T0) / D3PM_STEPS
    return [D3PM_T0 + i * step for i in range(D3PM_STEPS)]


# ======================================================================
# numpy 实现 GAME modules/functional.py + inference/utils.py
# ======================================================================
def format_boundaries(durations: list[float], length: int, timestep: float) -> np.ndarray:
    """累积时长 → 帧边界布尔数组（与 GAME format_boundaries 一致）。"""
    boundaries = np.zeros(length, dtype=bool)
    if not durations:
        return boundaries
    cum = np.cumsum(np.asarray(durations, dtype=np.float64))
    idx = np.rint(cum / timestep).astype(np.int64)
    for j in idx[:-1]:                      # 最后一个累计（=总长）丢弃
        if 0 <= j < length:
            boundaries[j] = True
    return boundaries


def boundaries_to_regions(boundaries: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
    regions = np.cumsum(boundaries.astype(np.int64), axis=-1) + 1
    if mask is not None:
        regions = regions * mask.astype(np.int64)
    return regions


def regions_to_durations(regions: np.ndarray, max_n: int) -> np.ndarray:
    """每个 region（1..max_n）的帧数。"""
    counts = np.zeros(max_n, dtype=np.int64)
    if max_n > 0:
        for r in regions:
            if 1 <= r <= max_n:
                counts[r - 1] += 1
    return counts


def parse_words(ph_seq: list[str], ph_dur: list[float], ph_num: list[int],
                uv_vocab: set[str] | None, uv_cond: str = "lead",
                merge_consecutive_uv: bool = False) -> tuple[list[float], list[int]]:
    """音素序列 + ph_num → 词时长 / 词 vuv（与 GAME parse_words 一致）。"""
    word_dur: list[float] = []
    word_vuv: list[int] = []
    idx = 0
    for num in ph_num:
        word_dur.append(float(sum(ph_dur[idx:idx + num])))
        vuv = 1
        if uv_vocab is not None:
            if uv_cond == "lead":
                if ph_seq[idx] in uv_vocab:
                    vuv = 0
            elif uv_cond == "all":
                if all(ph in uv_vocab for ph in ph_seq[idx:idx + num]):
                    vuv = 0
        word_vuv.append(vuv)
        idx += num
    if merge_consecutive_uv:
        merged_dur = [word_dur[0]] if word_dur else []
        merged_vuv = [word_vuv[0]] if word_vuv else []
        for dur, vuv in zip(word_dur[1:], word_vuv[1:]):
            if vuv == 0 and merged_vuv[-1] == 0:
                merged_dur[-1] += dur
            else:
                merged_dur.append(dur)
                merged_vuv.append(vuv)
        return merged_dur, merged_vuv
    return word_dur, word_vuv


def align_notes_to_words(word_dur: list[float], word_vuv: list[int],
                         note_seq: list, note_dur: list[float],
                         tol: float = 0.01, apply_word_uv: bool = False
                         ) -> tuple[list, list[float], list[int]]:
    """把音符序列吸附到词边界（与 GAME align_notes_to_words 一致）。

    note_seq 元素可以是音符名字符串（"C#4+30"）或直接是 float MIDI：
    本实现对两者等价处理（仅做相等合并与透传），因此调用方可以跳过
    「音符名」中间表示，直接传 float MIDI，避免音分在字符串化时丢失。
    """
    if not word_dur or not note_seq:
        return [], [], []
    word_start = np.cumsum([0.0] + word_dur[:-1])
    word_end = np.cumsum(word_dur)
    note_start = np.cumsum([0.0] + note_dur[:-1])
    note_end = np.cumsum(note_dur)

    new_seq: list = []
    new_dur: list[float] = []
    new_slur: list[int] = []
    for wi in range(len(word_dur)):
        if apply_word_uv and word_vuv[wi] == 0:
            new_seq.append("rest")
            new_dur.append(word_dur[wi])
            new_slur.append(0)
            continue
        ns = int(np.argmin(np.abs(note_start - word_start[wi])))
        if word_start[wi] < note_start[ns] - tol:
            ns = max(0, ns - 1)
        ne = int(np.argmin(np.abs(note_end[ns:] - word_end[wi])) + ns)
        if word_end[wi] > note_end[ne] + tol:
            ne = min(len(note_end) - 1, ne + 1)

        w_seq: list = []
        w_dur: list[float] = []
        for ni in range(ns, ne + 1):
            start = word_start[wi] if ni == ns else note_start[ni]
            end = word_end[wi] if ni == ne else note_end[ni]
            if w_seq and _same_pitch(w_seq[-1], note_seq[ni]):
                w_dur[-1] += (end - start)
            else:
                w_seq.append(note_seq[ni])
                w_dur.append(float(end - start))
        new_seq.extend(w_seq)
        new_dur.extend(w_dur)
        new_slur.extend([0] + [1] * (len(w_seq) - 1))
    return new_seq, new_dur, new_slur


def _same_pitch(a, b) -> bool:
    """合并判定：同名或浮点 midi 差 < 1 音分。"""
    if isinstance(a, str) or isinstance(b, str):
        return a == b
    return abs(float(a) - float(b)) < 0.01


# ======================================================================
# 时间轴压缩 / 还原
# GAME 的 encoder 用 `duration` 截断音频前端（mask = 前 L 帧），因此模型
# 的时间轴 = 只含音素时长、去掉音素间静音空隙的「压缩时间轴」。标注数据的
# TextGrid 常把词间静音留成未标注空隙，必须先按音素区间拼接音频，再把音符
# 时间映射回真实音频时间，否则出现整体漂移。
# ======================================================================
def condense_audio(audio: np.ndarray, sr: int,
                   phones: list[tuple[float, float, str]]
                   ) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """按音素区间拼接音频，去掉音素间/首尾空隙。返回 (音频, 压缩起点, 真实起点, 总长)。"""
    segs: list[np.ndarray] = []
    cond_starts: list[float] = []
    real_starts: list[float] = []
    n = len(audio)
    acc = 0
    for s, e, _ in phones:
        i0 = max(0, min(int(round(s * sr)), n))
        i1 = max(i0, min(int(round(e * sr)), n))
        cond_starts.append(acc / sr)
        real_starts.append(float(s))
        segs.append(audio[i0:i1])
        acc += i1 - i0
    if not segs:
        return np.zeros(0, dtype=np.float32), np.zeros(1), np.zeros(1), 0.0
    return (np.concatenate(segs).astype(np.float32),
            np.asarray(cond_starts, dtype=np.float64),
            np.asarray(real_starts, dtype=np.float64),
            acc / sr)


class CondTimeMap:
    """压缩时间轴 → 真实音频时间的分段线性映射。

    压缩轴相邻音素紧邻，真实轴音素间可能隔有空隙（未标注的静音）。因此处于
    音素边界的时间点在两个轴上含义不同，必须区分「作为起点」与「作为终点」：

      - 起点（含词首音素）：取该音素起点的真实时间
      - 终点（含词末音素）：取该音素终点的真实时间，绝不越过其后空隙

    否则落在词边界的音符端点会被映射到下一个词的真实起点，把空隙算进音符时长
    （曾出现 note 总时长比音素总时长多出空隙长度的问题）。
    """

    def __init__(self, ph_dur: list[float], cond_starts: np.ndarray,
                 real_starts: np.ndarray):
        self.cond_starts = np.asarray(cond_starts, dtype=np.float64)
        self.real_starts = np.asarray(real_starts, dtype=np.float64)
        self.ph_dur = np.asarray(ph_dur, dtype=np.float64)
        self.cond_ends = self.cond_starts + self.ph_dur

    def _clamp(self, i: int) -> int:
        return max(0, min(int(i), len(self.cond_starts) - 1))

    def to_real_start(self, t: float) -> float:
        i = self._clamp(np.searchsorted(self.cond_ends, t, side="right"))
        return float(self.real_starts[i] + max(0.0, t - self.cond_starts[i]))

    def to_real_end(self, t: float) -> float:
        i = self._clamp(np.searchsorted(self.cond_starts, t, side="left") - 1)
        off = max(0.0, min(t - self.cond_starts[i], float(self.ph_dur[i])))
        return float(self.real_starts[i] + off)


# ======================================================================
# GAME ONNX 对齐器
# ======================================================================
class GameNoteAligner:
    """GAME ONNX 词级音符生成。会话懒加载并复用。"""

    def __init__(self, model_dir: str | pathlib.Path, prefer_gpu: bool = True):
        self.model_dir = pathlib.Path(model_dir)
        self.prefer_gpu = prefer_gpu
        self._config: dict | None = None
        self._encoder = None
        self._segmenter = None
        self._estimator = None

    def _ensure(self):
        if self._encoder is not None:
            return
        cfg_path = self.model_dir / "config.json"
        self._config = json.loads(cfg_path.read_text(encoding="utf8"))
        self._timestep = float(self._config.get("timestep", GAME_TIMESTEP))
        self._languages: dict[str, int] = self._config.get("languages") or {}
        self._encoder = make_session(self.model_dir / "encoder.onnx", self.prefer_gpu)
        self._segmenter = make_session(self.model_dir / "segmenter.onnx", self.prefer_gpu)
        self._estimator = make_session(self.model_dir / "estimator.onnx", self.prefer_gpu)

    # ---- 内部：一步 D3PM 边界解码 -------------------------------------
    def _segment(self, x_seg: np.ndarray, known: np.ndarray, maskT: np.ndarray,
                 lang_id: int) -> np.ndarray:
        radius = np.array(int(round(SEG_RADIUS_SEC / self._timestep)), dtype=np.int64)
        threshold = np.array(SEG_THRESHOLD, dtype=np.float32)
        language = np.array([lang_id], dtype=np.int64)
        boundaries = known.copy()
        for ti in d3pm_ts():
            out = self._segmenter.run(["boundaries"], {
                "x_seg": x_seg,
                "language": language,
                "known_boundaries": known[None, :],
                "prev_boundaries": boundaries[None, :],
                "t": np.array([ti], dtype=np.float32),
                "maskT": maskT,
                "threshold": threshold,
                "radius": radius,
            })[0]
            boundaries = out[0].astype(bool)
        return boundaries

    def align(self, samples: np.ndarray, sr: int,
              phones: list[tuple[float, float, str]], ph_num: list[int],
              uv_vocab: set[str] | None = None, lang: str = "zh") -> list[Note]:
        """
        返回按词对齐后的音符列表（含 rest），时间为**真实音频时间**。

        phones: [(start, end, mark)]；ph_num: 词划分（同 add_ph_num，音素数）。
        """
        self._ensure()
        ph_seq = [m for _, _, m in phones]
        ph_dur = [float(e - s) for s, e, _ in phones]
        if not ph_seq or not ph_dur or not ph_num:
            return []

        # 1) 压实音频到音素时间轴（去掉词间/首尾静音空隙）
        cond_audio, cond_starts, real_starts, total = condense_audio(samples, sr, phones)
        if total <= 0 or len(cond_audio) == 0:
            return []
        if sr != GAME_SR:                     # 编码器按 44.1k 训练
            cond_audio = resample(cond_audio, sr, GAME_SR)

        # 2) 词时长：known 用「合并连续无声词」；对齐用未合并（与 GAME 一致）
        merge = uv_vocab is not None
        word_dur_known, _ = parse_words(ph_seq, ph_dur, ph_num, uv_vocab,
                                        uv_cond=UV_WORD_COND, merge_consecutive_uv=merge)
        word_dur_align, word_vuv = parse_words(ph_seq, ph_dur, ph_num, uv_vocab,
                                               uv_cond=UV_WORD_COND, merge_consecutive_uv=False)
        if not word_dur_known:
            return []

        # 3) 编码器：压缩音频 + 压缩总时长
        waveform = cond_audio[None, :].astype(np.float32)
        duration = np.array([total], dtype=np.float32)
        x_seg, x_est, maskT = self._encoder.run(
            ["x_seg", "x_est", "maskT"], {"waveform": waveform, "duration": duration})
        T = int(maskT.shape[1])

        # 4) 边界解码（已知词边界 + D3PM 采样）
        known = format_boundaries(word_dur_known, T, self._timestep) & maskT[0]
        lang_id = int(self._languages.get(lang, 0))
        boundaries = self._segment(x_seg, known, maskT, lang_id)

        regions = boundaries_to_regions(boundaries, maskT)[0]
        max_n = int(regions.max()) if regions.size else 0
        if max_n <= 0:
            return []

        # 5) 音符时值 / 音高 / 发声
        maskN = (np.arange(max_n) < max_n)[None, :]
        presence, scores = self._estimator.run(
            ["presence", "scores"], {
                "x_est": x_est,
                "boundaries": boundaries[None, :],
                "maskT": maskT,
                "maskN": maskN,
                "threshold": np.array(EST_THRESHOLD, dtype=np.float32),
            })
        presence = presence[0].astype(bool)
        scores = scores[0].astype(np.float32)

        durations = regions_to_durations(regions, max_n) * self._timestep
        valid = durations > 0
        note_dur = durations[valid].astype(np.float64)
        note_midi = scores[valid].astype(np.float64)
        note_vuv = presence[valid]
        if len(note_dur) == 0:
            return []

        # 6) 直接以 float MIDI 参与词对齐（不经过音符名字符串）。
        #    若走 _midi_to_note_name → _note_name_to_midi 往返，
        #    而 librosa.note_to_midi 不解析 "+cents" 后缀，音分被静默丢弃，
        #    会导致所有音符都落在整数半音上。
        note_seq = ["rest" if not v else float(m)
                    for m, v in zip(note_midi, note_vuv)]

        # 7) 吸附到词边界（压缩时间轴）
        seq, dur, slur = align_notes_to_words(
            word_dur_align, word_vuv, note_seq, note_dur.tolist(),
            apply_word_uv=(UV_NOTE_COND == "follow"))

        # 8) 压缩时间 → 真实音频时间
        tmap = CondTimeMap(ph_dur, cond_starts, real_starts)
        notes: list[Note] = []
        cursor = 0.0
        for item, d, s in zip(seq, dur, slur):
            end = cursor + d
            if d > 0:
                real_s, real_e = tmap.to_real_start(cursor), tmap.to_real_end(end)
                if item == "rest":
                    notes.append(Note(real_s, real_e, 0.0, slur=bool(s), rest=True))
                else:
                    notes.append(Note(real_s, real_e,
                                      round(float(item), 2),
                                      slur=bool(s), rest=False))
            cursor = end
        return notes
