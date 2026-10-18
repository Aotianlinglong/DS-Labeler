# -*- coding: utf-8 -*-
"""
每句工作数据：`.dsl/<句名>.json`。

内容 = 音素工作副本（来自 TextGrid，编辑不回写源 TG 除非手动"覆盖导出"）
     + 音符（首次由 GAME 生成，之后为用户编辑结果）
     + words / texts 两层（可人工编辑：拖边界 + 改名；首打开从源 TG 播种）。

words 层名的自动更新基线 `words_ref` 也随句落盘（见 core.edit.sync_word_marks）。

首打开：从 TextGrid 播种 phones（+ words/texts）+ 跑 GAME 得 notes → 保存；
再次打开：直接读取，秒载。
"""
from __future__ import annotations

import json
import pathlib
from dataclasses import dataclass, field

from core.notes import Note

FORMAT_VERSION = 2      # v2 起工作文件带 words / texts / words_ref（v1 旧文件仍可读）


@dataclass
class SentenceData:
    name: str
    phones: list[tuple[float, float, str]] = field(default_factory=list)
    notes: list[Note] = field(default_factory=list)
    # words / texts 层：[(start, end, mark)]，与 phones 同时间轴，可人工编辑
    words: list[tuple[float, float, str]] = field(default_factory=list)
    texts: list[tuple[float, float, str]] = field(default_factory=list)
    # 每个 word「上次同步时的音素原名列表」——word 名自动更新的基线（长度随 words）
    words_ref: list[list[str]] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "version": FORMAT_VERSION,
            "name": self.name,
            "phones": [[round(s, 6), round(e, 6), m] for s, e, m in self.phones],
            "notes": [n.to_dict() for n in self.notes],
            "words": [[round(s, 6), round(e, 6), m] for s, e, m in self.words],
            "texts": [[round(s, 6), round(e, 6), m] for s, e, m in self.texts],
            "words_ref": [[str(x) for x in g] for g in self.words_ref],
        }

    @classmethod
    def from_dict(cls, d: dict) -> "SentenceData":
        phones = [(float(s), float(e), str(m)) for s, e, m in d.get("phones", [])]
        notes = [Note.from_dict(nd) for nd in d.get("notes", [])]
        words = [(float(s), float(e), str(m)) for s, e, m in d.get("words", [])]
        texts = [(float(s), float(e), str(m)) for s, e, m in d.get("texts", [])]
        words_ref = [[str(x) for x in g] for g in d.get("words_ref", [])]
        return cls(name=d.get("name", ""), phones=phones, notes=notes,
                   words=words, texts=texts, words_ref=words_ref)

    def save(self, path: str | pathlib.Path) -> None:
        path = pathlib.Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), ensure_ascii=False, indent=1),
                        encoding="utf8")

    @classmethod
    def load(cls, path: str | pathlib.Path, name: str = "") -> "SentenceData":
        d = json.loads(pathlib.Path(path).read_text(encoding="utf8"))
        sd = cls.from_dict(d)
        if not sd.name:
            sd.name = name
        return sd
