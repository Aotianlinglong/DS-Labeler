# -*- coding: utf-8 -*-
"""数据模型：音符（钢琴窗）与音素（TextGrid 工作副本）。"""
from __future__ import annotations


class Note:
    """一个音符：时间区间 + MIDI 音高（浮点，保留 GAME 的 cents）+ slur + rest。

    midi 用浮点以保留模型给的音分（真实数据集 note_seq 形如 `A#3+30`）；
    界面显示/编辑取整，导出时按 cents 还原。
    """

    __slots__ = ("start", "end", "midi", "slur", "rest")

    def __init__(self, start: float, end: float, midi: float,
                 slur: bool = False, rest: bool = False):
        self.start = float(start)
        self.end = float(end)
        self.midi = float(midi)
        self.slur = bool(slur)
        self.rest = bool(rest)

    @property
    def dur(self) -> float:
        return self.end - self.start

    @property
    def midi_int(self) -> int:
        """取整 MIDI（界面显示 / 编辑用）。"""
        return int(round(self.midi))

    def note_name(self) -> str:
        """导出用音符名（rest 或 `C4` / `A#3+30`）。"""
        if self.rest:
            return "rest"
        import librosa
        return librosa.midi_to_note(self.midi, unicode=False, cents=True)

    def to_dict(self) -> dict:
        return {"start": round(self.start, 6), "end": round(self.end, 6),
                "midi": round(self.midi, 4), "slur": self.slur, "rest": self.rest}

    @classmethod
    def from_dict(cls, d: dict) -> "Note":
        return cls(d["start"], d["end"], d["midi"],
                   d.get("slur", False), d.get("rest", False))

    def __repr__(self) -> str:
        tag = " rest" if self.rest else ""
        tag += " slur" if self.slur else ""
        return f"Note({self.start:.3f}-{self.end:.3f}, midi={self.midi}{tag})"


def notes_to_tuples(notes: list[Note], skip_rest: bool = True
                    ) -> list[tuple[float, float, float, bool]]:
    """转成视图需要的 (start, end, midi, slur) 列表（midi 为浮点）。"""
    return [(n.start, n.end, n.midi, n.slur) for n in notes
            if not (skip_rest and n.rest)]
