# -*- coding: utf-8 -*-
"""
旧数据集（翻新）csv 读写：`transcriptions.csv` ↔ SentenceData。

翻新工程的数据目录是**上游流水线的领地**，本工具**只读**它（不往目录里写任何东西），
自己的工程文件与 F0 缓存一律落在软件根目录的 `refurbish/`（见 core.project）。

  - 输入：一个数据目录（自动探测 `wavs/` 或 `wav/` 或根下 *.wav + `transcriptions.csv`）
  - 层：**只有 phones 层**（旧数据无 words/texts；不做重新自动 GAME）
  - 音符：直接取 csv 的 note_seq / note_dur / note_slur（不做 GAME 推理）
  - 输出：只写一个新 csv，不写 TextGrid、不拷 wav

ph_num 由音素元辅分类（二分式）重算；若所选主语言是多音节系统，
会提示「无 words 层 → ph_num 退化为二分式」。
"""
from __future__ import annotations

import csv
import pathlib
import re

from core.export import ExportValidationError, notes_to_columns, validate_transcription

CSV_FIELDS = ["name", "ph_seq", "ph_dur", "ph_num", "note_seq", "note_dur", "note_slur"]
_NOTE_RE = re.compile(r"^([A-Ga-g])(#|b)?(-?\d+)([+-]\d+)?$")


def parse_note_token(tok: str) -> tuple[float | None, bool]:
    """`C4` / `A#3+30` / `rest` → (midi 浮点, 是否休止)。

    音分后缀必须保留（librosa.note_to_midi 不解析音分，故拆出后缀自行换算）。
    """
    tok = (tok or "").strip()
    if not tok or tok.lower() == "rest":
        return 0.0, True
    m = _NOTE_RE.match(tok)
    if not m:
        return None, False
    letter, acc, octv, cents = m.groups()
    import librosa
    try:
        midi = float(librosa.note_to_midi(letter.upper() + (acc or "") + octv))
    except Exception:
        return None, False
    if cents:
        midi += float(cents) / 100.0
    return midi, False


def make_classifier(main_lang: str = "zh"):
    """按主语言建分类器（四个语言都读内置词典，与工程分类器同源）。

    持有工程时优先用 `Project.make_classifier()`（可带自定义 symbols/systems）；
    这个入口供**不持有工程**的场合（单独校验 / 导出脚本）使用。
    """
    from core.paths import lang_dict
    from core.phonemes import LANGS, LanguageTable, PhonemeClassifier
    tables = {}
    for lang in LANGS:
        t = LanguageTable(lang)
        t.load_from_files(*lang_dict(lang))     # assets/dictionaries/<lang>/
        tables[lang] = t
    return PhonemeClassifier(main_lang=main_lang, tables=tables)


def detect_source(root: pathlib.Path) -> tuple[pathlib.Path | None, pathlib.Path | None]:
    """在目录中探测 (transcriptions.csv, wav 目录)。支持 `raw/` 一层嵌套。"""
    root = pathlib.Path(root)
    for base in (root, root / "raw"):
        if not base.is_dir():
            continue
        c = base / "transcriptions.csv"
        for wd in (base / "wavs", base / "wav", base):
            if wd.is_dir() and any(wd.glob("*.wav")):
                return (c if c.exists() else None), wd
    return None, None


def read_csv_names(csv_path: pathlib.Path) -> list[str]:
    """只读 csv 的 name 列（列表刷新用，避免整表重新解析）。"""
    out: list[str] = []
    with open(csv_path, "r", encoding="utf8", newline="") as f:
        for ri, row in enumerate(csv.DictReader(f)):
            out.append((row.get("name") or "").strip() or f"row{ri + 1}")
    return out


def read_csv_items(csv_path: pathlib.Path) -> list:
    """读旧数据集 csv → SentenceData 列表（phones 由 ph_seq/ph_dur 累积推区间）。

    note_seq 支持音分（`A#3+30`）与 `rest`；note_slur 缺失时按 0 处理。
    """
    from core.notes import Note
    from core.sentence import SentenceData

    out = []
    with open(csv_path, "r", encoding="utf8", newline="") as f:
        for ri, row in enumerate(csv.DictReader(f)):
            name = (row.get("name") or "").strip() or f"row{ri + 1}"
            ph = (row.get("ph_seq") or "").split()
            durs = [float(x) for x in (row.get("ph_dur") or "").split()]
            phones, t = [], 0.0
            for m, d in zip(ph, durs):
                phones.append((t, t + d, m))
                t += d
            nseq = (row.get("note_seq") or "").split()
            ndur = [float(x) for x in (row.get("note_dur") or "").split()]
            nslur = (row.get("note_slur") or "").split()
            notes, t = [], 0.0
            for i, (tok, d) in enumerate(zip(nseq, ndur)):
                midi, rest = parse_note_token(tok)
                if midi is None:                      # 无法解析 → 当休止，保证时间轴完整
                    midi, rest = 0.0, True
                slur = (nslur[i] == "1") if i < len(nslur) else False
                notes.append(Note(t, t + d, midi, slur=slur, rest=rest))
                t += d
            out.append(SentenceData(name=name, phones=phones, notes=notes))
    return out


def write_csv(out_path: pathlib.Path, items: list, classifier,
              warnings_out: dict | None = None) -> pathlib.Path:
    """写出新 csv（列序与主流程一致）。写盘前按 convert_ds 硬规则逐句校验，不过则拒写。"""
    out_path = pathlib.Path(out_path)
    rows, problems = [], {}
    for it in items:
        ph_seq = [m for _, _, m in it.phones]
        ph_dur = [e - s for s, e, _ in it.phones]
        ph_num = classifier.compute_ph_num(ph_seq)      # 无 words 层 → 二分式
        n_seq, n_dur, n_slur = notes_to_columns(it.notes)
        row = {
            "name": it.name,
            "ph_seq": " ".join(ph_seq),
            "ph_dur": " ".join(str(round(d, 6)) for d in ph_dur),
            "ph_num": " ".join(str(x) for x in ph_num),
            "note_seq": " ".join(n_seq),
            "note_dur": " ".join(n_dur),
            "note_slur": " ".join(n_slur),
        }
        errs, warns = validate_transcription(
            row["ph_seq"].split(), [float(x) for x in row["ph_dur"].split()],
            [int(x) for x in row["ph_num"].split()], row["note_seq"].split(),
            [float(x) for x in row["note_dur"].split()], row["note_slur"].split())
        if errs:
            problems[it.name] = errs
        if warns and warnings_out is not None:
            warnings_out[it.name] = warns
        rows.append(row)
    if problems:
        raise ExportValidationError(problems)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        w.writeheader()
        w.writerows(rows)
    return out_path
