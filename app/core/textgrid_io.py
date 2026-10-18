# -*- coding: utf-8 -*-
"""
TextGrid 读写：texts / words / phones 三层。
  - phones 可编辑（DS-Labeler 的工作对象，保留语言前缀）
  - texts / words 只读参照（导出时由 phones 反推 words）
"""
from __future__ import annotations

import pathlib
from dataclasses import dataclass, field

from textgrid import TextGrid


@dataclass
class TierData:
    name: str
    marks: list[str] = field(default_factory=list)
    xmins: list[float] = field(default_factory=list)
    xmaxs: list[float] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.marks)


@dataclass
class SentenceData:
    """一句音频的标注数据（从 TextGrid 解析）"""
    name: str                      # 句名（= wav 文件名 stem）
    texts: TierData = field(default_factory=lambda: TierData("texts"))
    words: TierData = field(default_factory=lambda: TierData("words"))
    phones: TierData = field(default_factory=lambda: TierData("phones"))

    @property
    def duration(self) -> float:
        return self.phones.xmaxs[-1] if self.phones.xmaxs else 0.0

    # ---- 便捷访问 ----------------------------------------------------
    def phone_marks(self) -> list[str]:
        return self.phones.marks

    def phone_durations(self) -> list[float]:
        return [b - a for a, b in zip(self.phones.xmins, self.phones.xmaxs)]

    def phone_start_times(self) -> list[float]:
        return self.phones.xmins


def parse_textgrid(tg_path: pathlib.Path, name: str | None = None) -> SentenceData:
    """解析 TextGrid → SentenceData。找不到层时报错提示。"""
    tg = TextGrid()
    tg.read(str(tg_path))
    data = SentenceData(name=name or pathlib.Path(tg_path).stem)
    for tier in tg:
        tier_data = TierData(name=tier.name)
        for interval in tier:
            tier_data.marks.append(interval.mark)
            tier_data.xmins.append(interval.minTime)
            tier_data.xmaxs.append(interval.maxTime)
        if tier.name == "texts":
            data.texts = tier_data
        elif tier.name == "words":
            data.words = tier_data
        elif tier.name == "phones":
            data.phones = tier_data
    if len(data.phones) == 0:
        raise ValueError(f"{tg_path.name}: 找不到 phones 层（或 phones 层为空）")
    return data


def build_textgrid(data: SentenceData, xmax: float | None = None) -> TextGrid:
    """从 SentenceData 构建 TextGrid（通用构造，供导出/回写使用）。"""
    from textgrid import IntervalTier

    tg = TextGrid()
    end = xmax if xmax is not None else (data.phones.xmaxs[-1] if data.phones.xmaxs else 0.0)
    for tier_data in (data.texts, data.words, data.phones):
        tier = IntervalTier(name=tier_data.name, minTime=0.0, maxTime=max(end, 0.001))
        for mark, x0, x1 in zip(tier_data.marks, tier_data.xmins, tier_data.xmaxs):
            tier.add(minTime=x0, maxTime=x1, mark=mark)
        tg.append(tier)
    return tg


def write_textgrid(tg: TextGrid, path: pathlib.Path) -> None:
    tg.write(str(path))


def _fnum(v: float) -> str:
    """Praat 文本里的时间格式（与源文件一致）：整数不带小数，否则 6 位小数。"""
    if abs(v - int(v)) < 1e-9:
        return str(int(v))
    return f"{v:.6f}"


def write_praat_textgrid(path: pathlib.Path, xmax: float,
                         tiers: list[tuple[str, list[tuple[float, float, str]]]]) -> None:
    """
    手写 Praat TextGrid（不依赖 textgrid 库的 write）。

    textgrid 库的 `write()` 会把层内空隙补成空标记区间，导致下游 build_dataset
    读出空音素；源 TG 是把空隙留成「缺失区间」。这里保持一致。
    """
    lines: list[str] = [
        'File type = "ooTextFile"',
        'Object class = "TextGrid"',
        "",
        "xmin = 0",
        f"xmax = {_fnum(xmax)}",
        "tiers? <exists>",
        f"size = {len(tiers)}",
        "item []:",
    ]
    for ti, (name, intervals) in enumerate(tiers, 1):
        ivs = sorted(intervals, key=lambda x: x[0])
        lines.append(f"    item [{ti}]:")
        lines.append('        class = "IntervalTier"')
        lines.append(f'        name = "{name}"')
        lines.append("        xmin = 0")
        lines.append(f"        xmax = {_fnum(xmax)}")
        lines.append(f"        intervals: size = {len(ivs)}")
        for ii, (a, b, m) in enumerate(ivs, 1):
            lines.append(f"        intervals [{ii}]:")
            lines.append(f"            xmin = {_fnum(a)}")
            lines.append(f"            xmax = {_fnum(b)}")
            lines.append(f'            text = "{m}"')
    pathlib.Path(path).write_text("\n".join(lines) + "\n", encoding="utf8")


def write_edited_textgrid(tg_path: pathlib.Path,
                          phones: list[tuple[float, float, str]],
                          words: list[tuple[float, float, str]] | None = None,
                          texts: list[tuple[float, float, str]] | None = None,
                          xmax: float | None = None) -> None:
    """
    用编辑后的三层（phones / words / texts）覆盖写出 TextGrid。

    words / texts 现在**由用户编辑**（拖边界 + 双击改名），这里直接落盘，
    不再由 phones 反推。words 为空时（罕见）退化为「每音素一词」；
    texts 为空时区间随 words、标记留空。空隙保持为「缺失区间」，不写空标记
    （与源格式一致，避免下游读出空音素）。
    """
    tg_path = pathlib.Path(tg_path)
    phones = [(float(s), float(e), str(m)) for s, e, m in phones]
    words = [(float(s), float(e), str(m)) for s, e, m in (words or [])]
    texts = [(float(s), float(e), str(m)) for s, e, m in (texts or [])]
    if not words:
        words = [(s, e, m) for s, e, m in phones]      # 退化：每音素一词
    if not texts:
        texts = [(s, e, "") for s, e, _m in words]
    ends = ([e for _s, e, _m in phones] + [e for _s, e, _m in words]
            + [e for _s, e, _m in texts])
    end = float(xmax) if xmax is not None else max([0.001] + ends)
    write_praat_textgrid(tg_path, end, [
        ("texts", texts),
        ("words", words),
        ("phones", phones),
    ])


# ======================================================================
# 工具：TierData → [(start, end, mark)]（用于把源 TG 的 words/texts 播种进工作文件）
# ======================================================================
def tier_items(tier: TierData | None) -> list[tuple[float, float, str]]:
    """把解析出的 TierData 转成 `[(start, end, mark)]`；None → 空表。"""
    if tier is None:
        return []
    return [(float(a), float(b), str(m))
            for a, b, m in zip(tier.xmins, tier.xmaxs, tier.marks)]

