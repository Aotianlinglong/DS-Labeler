# -*- coding: utf-8 -*-
"""
导出闭环：`raw/wavs/*.wav`（44.1k mono 副本）+ `raw/transcriptions.csv`。

CSV 列与现役流水线（build_dataset.py + add_ph_num.py）保持一致：
  name, ph_seq, ph_dur, ph_num, note_seq, note_dur, note_slur
- ph_dur 取音素区间时长（秒，绝对时间轴，与 build_dataset --skip_silence_insertion 一致）
- ph_num 由音素元辅分类重算（与 add_ph_num 算法一致）
- note_seq 为 MIDI 整数，休止（rest）记 0
"""
from __future__ import annotations

import csv
import pathlib

import soundfile as sf

from core.notes import Note
from core.paths import lang_dict
from core.phonemes import PhonemeClassifier
from core.sentence import SentenceData

SAMPLE_RATE = 44100  # 固定 44.1k mono（与 build_dataset 输出一致）

# 词边界↔音符边界的容差（与 convert_ds.py 的 --tolerance 默认值一致）
WORD_NOTE_TOL = 0.005


class ExportValidationError(Exception):
    """导出前校验失败：会把问题按句名汇总，**拒写文件**，绝不产出不合规 CSV。"""

    def __init__(self, problems: dict[str, list[str]]):
        self.problems = problems
        super().__init__("导出数据未通过下游规则校验")


def validate_transcription(ph_seq: list[str], ph_dur: list[float], ph_num: list[int],
                           note_seq: list[str], note_dur: list[float],
                           note_slur: list[str],
                           tol: float = WORD_NOTE_TOL) -> tuple[list[str], list[str]]:
    """按下游 convert_ds.py 的真实规则校验一行，返回 `(硬错误, 警告)`。

    **硬错误** = convert_ds.py 的 assert，命中即下游直接崩（必须拒写）：
      E1 len(ph_dur) == sum(ph_num)                      （convert_ds.py:150）
      E2 len(note_seq) == len(note_dur)（== len(note_slur)）（convert_ds.py:151/81）
      E3 |Σph_dur − Σnote_dur| ≤ tol（默认 0.005）        （convert_ds.py:154）

    **警告** = 仅当 CSV **不写** note_slur 时下游才会要求的对齐条件。
    本工具总是写出 note_slur（convert_ds.py:159-160 会直接采用该列、不再走匹配），
    故这些不阻断导出，但出现时说明「词-音符」对应关系可疑，需人工确认：
      W1 len(ph_num) > len(note_dur)  —— 有词没有起始音符
      W2 词边界没落在音符边界上（±tol）—— 词与音符的时间切分错位
    """
    errs: list[str] = []
    warns: list[str] = []
    if len(ph_seq) != len(ph_dur) or sum(ph_num) != len(ph_seq):
        errs.append(f"音素数不匹配：音素 {len(ph_seq)} / 时长 {len(ph_dur)} / "
                    f"Σph_num {sum(ph_num)}（len(ph_dur) 必须等于 Σph_num）")
    if not (len(note_seq) == len(note_dur) == len(note_slur)):
        errs.append(f"音符三列长度不一致：seq {len(note_seq)} / dur {len(note_dur)} / "
                    f"slur {len(note_slur)}")
    if ph_dur and note_dur:
        _dd = sum(ph_dur) - sum(note_dur)
        if abs(_dd) > tol:
            errs.append(f"总时长不一致：音素 {sum(ph_dur):.4f}s vs 音符 {sum(note_dur):.4f}s"
                        f"（差 {_dd:+.4f}s，容差 {tol:.3f}s）")
    if len(ph_num) > len(note_dur):
        warns.append(f"词数 {len(ph_num)} 多于音符数 {len(note_dur)}：有词没有起始音符"
                     "（拆音素后需在 MIDI 上补对应音符）")
    word_pos, idx = [], 0
    for w in ph_num:
        if idx + w > len(ph_dur):       # E1 已失败时不再继续推算
            break
        word_pos.append((word_pos[-1] if word_pos else 0.0) + sum(ph_dur[idx:idx + w]))
        idx += w
    note_pos, nacc = [], 0.0
    for d in note_dur:
        nacc += d
        note_pos.append(nacc)
    if note_pos and word_pos:
        misses = [w for w in word_pos if min(abs(w - n) for n in note_pos) > tol]
        if misses:
            warns.append(f"{len(misses)} 个词边界没有对齐的音符边界"
                         f"（首个在 {misses[0]:.3f}s，容差 {tol:.3f}s）")
    return errs, warns


def notes_to_columns(notes: list[Note]) -> tuple[list[str], list[str], list[str]]:
    """音符列表 → note_seq / note_dur / note_slur 三列。

    格式与现役流水线（GAME 的 UpdateDiffSingerTranscriptionsCallback）逐字符一致：
      - note_seq：休止记 `rest`，其余为 librosa.midi_to_note(cents=True) 的音名（如 `A#3+30`）
      - note_dur：3 位小数（`f"{dur:.3f}"`）
      - note_slur：`0` / `1`
    """
    seq, dur, slur = [], [], []
    for n in sorted(notes, key=lambda x: (x.start, x.end)):
        seq.append(n.note_name())
        dur.append(f"{n.dur:.3f}")
        slur.append("1" if n.slur else "0")
    return seq, dur, slur


def _write_wav(src: pathlib.Path, dst: pathlib.Path, samplerate: int) -> None:
    data, sr = sf.read(str(src), dtype="float32", always_2d=True)
    mono = data.mean(axis=1) if data.shape[1] > 1 else data[:, 0]
    if sr != samplerate:
        import librosa
        mono = librosa.resample(mono, orig_sr=sr, target_sr=samplerate)
    sf.write(str(dst), mono, samplerate)


def export_dataset(out_dir: str | pathlib.Path,
                   items: list[SentenceData],
                   classifier: PhonemeClassifier,
                   wav_src_dir: str | pathlib.Path,
                   write_wavs: bool = True,
                   samplerate: int = SAMPLE_RATE,
                   word_sizes_by_name: dict[str, list[int]] | None = None,
                   warnings_out: dict[str, list[str]] | None = None) -> pathlib.Path:
    """导出数据集，返回 transcriptions.csv 路径。

    word_sizes_by_name：每句的「每词音素数」（由 words 层反推），
    **仅多音节音素系统**需要它；二分式/单音节忽略它以保证与原流水线逐字符一致。

    写文件前按 convert_ds.py 的硬规则逐句校验（validate_transcription）：
    有硬错误 → raise ExportValidationError，**不写任何产物**；
    软问题（词/音符切分可疑）→ 收集到 warnings_out（不阻断，由调用方提示）。
    """
    out_dir = pathlib.Path(out_dir)
    wav_src_dir = pathlib.Path(wav_src_dir)

    rows = []
    problems: dict[str, list[str]] = {}
    for it in items:
        ph_seq = [m for _, _, m in it.phones]
        ph_dur = [e - s for s, e, _ in it.phones]
        if not ph_seq:
            continue
        ph_num = classifier.compute_ph_num(ph_seq, (word_sizes_by_name or {}).get(it.name))
        n_seq, n_dur, n_slur = notes_to_columns(it.notes)
        row = {
            "name": it.name,
            "ph_seq": " ".join(ph_seq),
            "ph_dur": " ".join(str(round(d, 6)) for d in ph_dur),   # 与 build_dataset 逐字符一致
            "ph_num": " ".join(str(x) for x in ph_num),
            "note_seq": " ".join(n_seq),
            "note_dur": " ".join(n_dur),
            "note_slur": " ".join(n_slur),
        }
        # 直接校验**将要写出的那一行**（parse 回来），与下游读到的值完全一致
        _errs, _warns = validate_transcription(
            row["ph_seq"].split(), [float(x) for x in row["ph_dur"].split()],
            [int(x) for x in row["ph_num"].split()], row["note_seq"].split(),
            [float(x) for x in row["note_dur"].split()], row["note_slur"].split())
        if _errs:
            problems[it.name] = _errs
        if _warns and warnings_out is not None:
            warnings_out[it.name] = _warns
        rows.append(row)
    if problems:
        raise ExportValidationError(problems)      # 先校验，再落盘

    wavs_dir = out_dir / "wavs"
    csv_path = out_dir / "transcriptions.csv"
    out_dir.mkdir(parents=True, exist_ok=True)
    if write_wavs:
        wavs_dir.mkdir(parents=True, exist_ok=True)
        for it in items:
            src = wav_src_dir / f"{it.name}.wav"
            if src.exists():
                _write_wav(src, wavs_dir / f"{it.name}.wav", samplerate)
    with open(csv_path, "w", encoding="utf8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "name", "ph_seq", "ph_dur", "ph_num", "note_seq", "note_dur", "note_slur"])
        writer.writeheader()
        writer.writerows(rows)
    return csv_path


def unknown_report(items: list[SentenceData],
                   classifier: PhonemeClassifier) -> list[str]:
    """收集未命中元辅表的音素（导出前提示）。"""
    seen: set[str] = set()
    out: list[str] = []
    for it in items:
        for _, _, m in it.phones:
            if classifier.classify(m) in ("unknown", "bad_lang") and m not in seen:
                seen.add(m)
                out.append(m)
    return out


__all__ = [
    "export_dataset", "notes_to_columns", "unknown_report",
    "SAMPLE_RATE", "lang_dict",
]
