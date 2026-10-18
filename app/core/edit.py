# -*- coding: utf-8 -*-
"""
编辑模型 + 编辑操作（纯 Python，无 Qt 依赖，便于离线验证）。

音素编辑：切分 / 复制 / 删除（连续模式语义）；
音符编辑：词级音符、拖音高不动边界、边界移动时端点吸附。

实现分两层：
  - `mut_*`   ：直接改 model 状态、不入撤销栈（供鼠标拖动中实时预览）
  - 公开函数   ：一次点击 = 一条可撤销命令（MutateCommand）
拖动结束由视图用 `SnapshotCommand` 把整段拖动合并成一条撤销记录。
"""
from __future__ import annotations

import copy

from core.notes import Note
from core.sentence import SentenceData
from core.undo import MutateCommand, SnapshotCommand, UndoStack

EPS = 1e-4
MIN_PHONE_DUR = 0.0116     # 音素最小时长（秒；= 1 帧 @44.1kHz/hop512 = 11.61ms，避免亚帧音素）
MIN_NOTE_DUR = 0.01        # 音符最小时长（秒）
PITCH_MIN, PITCH_MAX = 12.0, 120.0   # MIDI 音高上下限
DEFAULT_PITCH = 60.0                 # 取消无声时的兜底音高（C4；正常情况会借用相邻音符）


class EditModel:
    """当前句的可编辑状态 + 撤销栈。"""

    def __init__(self, data: SentenceData):
        self.data = data
        self.name = data.name
        self.stack = UndoStack()
        self.dirty = False

    # ---- 数据存取 -----------------------------------------------------
    @property
    def phones(self) -> list[tuple[float, float, str]]:
        return self.data.phones

    @phones.setter
    def phones(self, v):
        self.data.phones = v

    @property
    def notes(self) -> list[Note]:
        return self.data.notes

    @notes.setter
    def notes(self, v):
        self.data.notes = v

    @property
    def words(self) -> list[tuple[float, float, str]]:
        return self.data.words

    @words.setter
    def words(self, v):
        self.data.words = v

    @property
    def texts(self) -> list[tuple[float, float, str]]:
        return self.data.texts

    @texts.setter
    def texts(self, v):
        self.data.texts = v

    def snapshot(self):
        # 快照覆盖**全部可编辑层**，这样撤销/重做能整体回滚（含 words/texts 与改名基线）
        return (list(self.data.phones), copy.deepcopy(self.data.notes),
                list(self.data.words), list(self.data.texts),
                copy.deepcopy(self.data.words_ref))

    def restore(self, snap) -> None:
        self.data.phones = list(snap[0])
        self.data.notes = copy.deepcopy(snap[1])
        self.data.words = list(snap[2])
        self.data.texts = list(snap[3])
        self.data.words_ref = copy.deepcopy(snap[4])

    # ---- 命令 ---------------------------------------------------------
    def run(self, label: str, fn) -> None:
        self.stack.push(MutateCommand(label, fn), self)

    def commit_snapshot(self, label: str, before) -> bool:
        """把「拖动前快照 → 当前状态」登记为一条撤销记录；无变化则不登记。"""
        after = self.snapshot()
        if _same(before, after):
            return False
        self.stack.push(SnapshotCommand(label, before, after), self)
        self.dirty = True
        return True

    def undo(self) -> str | None:
        return self.stack.undo(self)

    def redo(self) -> str | None:
        return self.stack.redo(self)

    # ---- 查询 ---------------------------------------------------------
    def phone_index_at(self, t: float) -> int:
        """t 落在哪个音素的 [start, end] 内；空隙则取最近的前一个。"""
        for i, (s, e, _) in enumerate(self.phones):
            if s - EPS <= t <= e + EPS:
                return i
        idx = -1
        for i, (s, e, _) in enumerate(self.phones):
            if e < t:
                idx = i
        return idx

    def boundary_near(self, t: float, tol: float) -> int | None:
        """容差内最近的边界 index（边界 i 位于 phone i-1 与 i 之间；i∈[1..n]）。"""
        best, best_d = None, tol
        n = len(self.phones)
        for i in range(1, n):
            d = abs(self.phones[i][0] - t)
            if d < best_d:
                best, best_d = i, d
        if n:
            d = abs(self.phones[-1][1] - t)
            if d < best_d:
                best = n
        return best

    def note_index_at(self, t: float, midi: float, tol_t: float = 0.0,
                      tol_midi: float = 0.6, include_rest: bool = False) -> int:
        for i, n in enumerate(self.notes):
            if n.rest and not include_rest:
                continue
            if n.start - tol_t <= t <= n.end + tol_t:
                if n.rest or abs(n.midi - midi) <= tol_midi:
                    return i
        return -1


def _same(a, b) -> bool:
    if (a[0] != b[0] or a[2] != b[2] or a[3] != b[3] or a[4] != b[4]
            or len(a[1]) != len(b[1])):
        return False
    for x, y in zip(a[1], b[1]):
        if (x.start, x.end, x.midi, x.slur, x.rest) != (y.start, y.end, y.midi, y.slur, y.rest):
            return False
    return True


# ======================================================================
# 音素：底层变更（mut_*）
# ======================================================================
def mut_split_phone(md: EditModel, index: int, position: float) -> bool:
    s, e, m = md.phones[index]
    if not (s + 1e-3 < position < e - 1e-3):
        return False
    md.phones[index] = (s, position, m)
    md.phones.insert(index + 1, (position, e, m))
    return True


def mut_duplicate_phone(md: EditModel, index: int) -> bool:
    s, e, m = md.phones[index]
    if e - s < 1e-3:
        return False
    mid = (s + e) / 2.0
    md.phones[index] = (s, mid, m)
    md.phones.insert(index + 1, (mid, e, m))
    return True


def mut_remove_phone(md: EditModel, index: int) -> bool:
    """Delete：删除音素；与前一音素连续时延长其 end（连续模式语义）。

    同时**同步音符层**（与 TG 同一套「吸收」逻辑），否则被删音素上的音符会
    悬空 —— 表现为 MIDI 上标签变空白 / 显示成 SP。
    """
    if not (0 <= index < len(md.phones)):
        return False
    removed = md.phones.pop(index)
    rs, re = float(removed[0]), float(removed[1])
    prev = index - 1
    contig = 0 <= prev < len(md.phones) and abs(md.phones[prev][1] - rs) < 1e-3
    if contig:
        s, _e, mk = md.phones[prev]
        md.phones[prev] = (s, re, mk)
    _heal_notes_after_remove(md, rs, re, contig)
    return True


def _heal_notes_after_remove(md: EditModel, rs: float, re: float,
                             contig_prev: bool) -> None:
    """删除音素后把音符层按 TG 的「吸收」语义同步好，杜绝孤儿音符 / 空白标签。

    - 前邻连续（TG 把 [rs, re] 并入前邻）且存在终止于 rs 的音符：
      该音符向后扩到 re（吸收），完全落在 [rs, re] 内的音符一并并入（删除）；
    - 其余情况（被删区间成为空隙，或找不到可吸收的前邻音符）：
      完全落在 [rs, re] 内的音符转为休止音符（R），时间轴保持完整。
    """
    notes = md.notes
    if not notes or re <= rs + 1e-6:
        return
    ins = {id(n) for n in notes if n.start >= rs - 1e-3 and n.end <= re + 1e-3}
    if not ins:
        return
    absorbed = False
    if contig_prev:
        for n in notes:
            if id(n) in ins:
                continue
            if abs(n.end - rs) < 1e-3:
                n.end = re
                absorbed = True
                break
    if absorbed:
        md.notes = [n for n in notes if id(n) not in ins]
    else:
        for n in notes:
            if id(n) in ins and not n.rest:
                n.rest, n.midi, n.slur = True, 0.0, False
    md.notes.sort(key=lambda x: (x.start, x.end))


def mut_rename_phone(md: EditModel, index: int, name: str) -> bool:
    if not (0 <= index < len(md.phones)):
        return False
    s, e, _ = md.phones[index]
    md.phones[index] = (s, e, name)
    return True


def mut_set_boundary(md: EditModel, index: int, new_t: float,
                     old_t: float | None = None, end_limit: float | None = None) -> bool:
    """按边界 index（1..n）改 phones 时间；**夹在相邻音素之间不穿越**。

    **向上联动**：words / texts 层里时间与 old_t 对齐的边界一起移到同一新时间
    （取各层可动范围交集后夹取 → 联动后仍严格对齐、不产生零长条目）。
    端点落在此边界上的音符端点吸附。
    """
    p = md.phones
    n = len(p)
    if n == 0 or index <= 0 or index > n:
        return False
    if old_t is None:
        old_t = p[index - 1][1]
    lo, hi = boundary_limits(md, index, end_limit)
    for items in (md.words, md.texts):
        l2, h2 = tier_move_limits(items, old_t)
        lo, hi = max(lo, l2), min(hi, h2)
    if hi < lo:
        hi = lo
    new_t = max(lo, min(hi, new_t))
    if index - 1 < n:
        s, _e, mk = p[index - 1]
        p[index - 1] = (s, new_t, mk)
    if index < n:
        _s, e, mk = p[index]
        p[index] = (new_t, e, mk)
    for items in (md.words, md.texts):
        _move_edges(items, old_t, new_t)
    for nt in md.notes:
        if abs(nt.start - old_t) < 1e-3:
            nt.start = new_t
        if abs(nt.end - old_t) < 1e-3:
            nt.end = new_t
    md.notes.sort(key=lambda x: (x.start, x.end))
    return True


# ======================================================================
# words / texts 层：边操作 + 向上联动（只有 phones→上层、words→texts）
# ======================================================================
def tier_edges_at(items, t: float, eps: float = EPS) -> list[tuple[int, str]]:
    """某层里时间 ≈ t 的所有边 `(条目下标, "start"|"end")`。

    按「条目边」而非「边界号」匹配 → 天然兼容层内空隙（缺失区间），且相邻条目的
    end 与其后条目的 start 会被一并返回（两条边都随动）。
    """
    out: list[tuple[int, str]] = []
    for i, (s, e, _m) in enumerate(items):
        if abs(float(s) - t) < eps:
            out.append((i, "start"))
        if abs(float(e) - t) < eps:
            out.append((i, "end"))
    return out


def _move_edges(items, t: float, new_t: float, eps: float = EPS) -> None:
    """把某层里时间 ≈ t 的边原地移到 new_t（就地改列表元素）。"""
    for idx, side in tier_edges_at(items, t, eps):
        s, e, m = items[idx]
        items[idx] = (new_t, e, m) if side == "start" else (s, new_t, m)


def tier_move_limits(items, t: float, eps: float = EPS) -> tuple[float, float]:
    """把某层里时间 ≈ t 的同组边**一起**移动时的可用范围 [lo, hi]。

    移动的边彼此绑定（相邻 end/start 同时动），约束只来自**不随动**的邻边，
    保证移动后每条目仍有正时长（用 MIN_PHONE_DUR 兜底最小间隔）。
    """
    lo, hi = float("-inf"), float("inf")
    for idx, side in tier_edges_at(items, t, eps):
        s, e, _m = items[idx]
        if side == "start":
            left_end = float(items[idx - 1][1]) if idx > 0 else None
            if left_end is None or abs(left_end - t) >= eps:
                lo = max(lo, left_end + MIN_PHONE_DUR if left_end is not None else float("-inf"))
            hi = min(hi, float(e) - MIN_PHONE_DUR)
        else:
            right_start = float(items[idx + 1][0]) if idx + 1 < len(items) else None
            if right_start is None or abs(right_start - t) >= eps:
                hi = min(hi, right_start - MIN_PHONE_DUR if right_start is not None else float("inf"))
            lo = max(lo, float(s) + MIN_PHONE_DUR)
    return lo, hi


def mut_move_tier_time(md: EditModel, tier: str, old_t: float, new_t: float) -> float | None:
    """拖动 words / texts 层里时间 ≈ old_t 的**一组边**（拖动中的实时预览，不入撤销栈）。

    同一时间上的相邻条目 end/start 一起动 → 该条边界整体平移，层内仍连续。
    - words 层：texts 层里同时间对齐的边一起移动；**不动 phones**。
    - texts 层：只动自己。
    可用范围取「本层 + （words 时的）texts 层」的交集后夹取。
    返回实际落点时间（夹取后）；该时间无边界可动时返回 None。
    """
    items = md.words if tier == "words" else md.texts
    if not tier_edges_at(items, old_t):
        return None
    lo, hi = tier_move_limits(items, old_t)
    if tier == "words":
        l2, h2 = tier_move_limits(md.texts, old_t)
        lo, hi = max(lo, l2), min(hi, h2)
    if hi < lo:
        hi = lo
    new_t = max(lo, min(hi, new_t))
    _move_edges(items, old_t, new_t)
    if tier == "words":
        _move_edges(md.texts, old_t, new_t)
    return new_t


def mut_duplicate_tier_item(md: EditModel, tier: str, index: int) -> bool:
    """Ctrl+D：条目从中点切两半、两块同名（与音素复制同一语义）。**不动下层**。"""
    items = md.words if tier == "words" else md.texts
    if not (0 <= index < len(items)):
        return False
    s, e, m = items[index]
    if e - s < 1e-3:
        return False
    mid = (s + e) / 2.0
    items[index] = (s, mid, m)
    items.insert(index + 1, (mid, e, m))
    if tier == "words":
        ref = md.data.words_ref
        if len(ref) == len(items) - 1:
            # 后半基线留空 → 下次 sync 按它自己的音素集合改名；前半沿用旧基线不变
            ref.insert(index + 1, [])
    return True


def mut_remove_tier_item(md: EditModel, tier: str, index: int) -> bool:
    """Delete：删除条目；与前一相邻条目连续时把区间并入前一条目（同音素删除语义）。

    **不动 phones / 音符**（words/texts 层是可独立编辑的层，不向下联动）。
    """
    items = md.words if tier == "words" else md.texts
    if not (0 <= index < len(items)):
        return False
    rs, re, _m = items[index]
    items.pop(index)
    prev = index - 1
    if 0 <= prev < len(items) and abs(float(items[prev][1]) - rs) < 1e-3:
        s, _e, mk = items[prev]
        items[prev] = (s, re, mk)
    if tier == "words":
        ref = md.data.words_ref
        if len(ref) == len(items) + 1:
            ref.pop(index)      # 基线同步删一项（并入前邻时前邻音素集变了 → 会按音素集合改名）
    return True


def mut_rename_tier_item(md: EditModel, tier: str, idx: int, name: str) -> bool:
    items = md.words if tier == "words" else md.texts
    if not (0 <= idx < len(items)):
        return False
    s, e, _m = items[idx]
    items[idx] = (s, e, name)
    return True


def duplicate_tier_item(model: EditModel, tier: str, index: int) -> bool:
    """Ctrl+D：words / texts 条目从中点切两半、两块同名（一次点击 = 一条可撤销命令）。"""
    items = model.words if tier == "words" else model.texts
    if not (0 <= index < len(items)) or items[index][1] - items[index][0] < 1e-3:
        return False
    model.run("复制词" if tier == "words" else "复制短语",
              lambda md: mut_duplicate_tier_item(md, tier, index))
    return True


def remove_tier_item(model: EditModel, tier: str, index: int) -> bool:
    """Delete：删除 words / texts 条目（与前一连续则并入前一条目）。"""
    items = model.words if tier == "words" else model.texts
    if not (0 <= index < len(items)):
        return False
    model.run("删除词" if tier == "words" else "删除短语",
              lambda md: mut_remove_tier_item(md, tier, index))
    return True


def rename_tier_item(model: EditModel, tier: str, idx: int, name: str) -> bool:
    """双击改名（一次点击 = 一条可撤销命令）。"""
    items = model.words if tier == "words" else model.texts
    if not (0 <= idx < len(items)) or items[idx][2] == name:
        return False
    label = "重命名词" if tier == "words" else "重命名短语"
    model.run(label, lambda md: mut_rename_tier_item(md, tier, idx, name))
    return True


# ======================================================================
# word ↔ 音素归属（替代旧的「phones 反推 words/texts」）
# ======================================================================
def _word_of(t: float, ws: list[float], we: list[float]) -> int:
    """时间 t 落在哪个 word：包含判定（边界同值时取**右侧**词）；全不包含则取最近词。"""
    cands = [k for k in range(len(ws)) if ws[k] - EPS <= t <= we[k] + EPS]
    if cands:
        return max(cands, key=lambda k: ws[k])
    return min(range(len(ws)), key=lambda k: min(abs(t - ws[k]), abs(t - we[k])))


def word_phone_groups(phones, words) -> list[list[int]]:
    """音素归给**其起点所落**的 word 的音素下标组（所有下游的唯一归属口径）。"""
    groups: list[list[int]] = [[] for _ in words]
    if not phones or not words:
        return groups
    ws = [float(w[0]) for w in words]
    we = [float(w[1]) for w in words]
    for i, (s, _e, _m) in enumerate(phones):
        groups[_word_of(float(s), ws, we)].append(i)
    return groups


def word_phone_counts(phones, words) -> list[int] | None:
    """每 word 音素数（供导出 ph_num 的多音节系统）；无 words 层 → None。

    空词（音素数为 0）被剔除，保证 Σ == 音素数（导出硬校验 E1）。
    """
    if not words:
        return None
    return [c for c in (len(g) for g in word_phone_groups(phones, words)) if c > 0]


def sync_word_marks(md: EditModel) -> bool:
    """word 名自动更新：某 word 下的音素**原名列表**与基线不同 → 改写为空格连接并更新基线；
    相同则**保持原样**（源 words 标记不被吞掉）。返回是否有改动。
    """
    words = md.words
    if not words:
        return False
    groups = word_phone_groups(md.phones, words)
    ref = list(md.data.words_ref)
    if len(ref) != len(words):
        ref = [list(ref[k]) if k < len(ref) else [] for k in range(len(words))]
    changed = False
    out = list(words)
    for k in range(len(words)):
        names = [md.phones[i][2] for i in groups[k]]
        if names != list(ref[k]):
            s, e, _m = out[k]
            out[k] = (s, e, " ".join(names))
            ref[k] = names
            changed = True
    md.data.words = out
    md.data.words_ref = ref
    return changed


def _ph_disp(classifier, ph: str) -> str:
    """音符标签里音素的显示名 = **本体**（`zh/x` → `x`、`en/ow` → `ow`），不带语言前缀。

    words 层名字不受影响（那边照旧写音素原名集合）。
    """
    return brief_phone(classifier, ph) if classifier is not None else (ph.partition("/")[2] or ph)


def note_word_labels(md: EditModel, classifier=None) -> list[str]:
    """有 words 层时的音符标签：非休止非连音音符 → 其**起点所在 word** 下音素的合集
    （如 `w uo vf`）。合规 TG 下一个 word 只对应一个非连音音符；若出现第 2、3…个 → 留空。
    休止 → `R`，连音 → `-`。音素只显示本体，**不带语言前缀**（`en/d en/ow en/n` → `d ow n`）。
    """
    words = md.words
    groups = word_phone_groups(md.phones, words)
    ws = [float(w[0]) for w in words]
    we = [float(w[1]) for w in words]
    shown: set[int] = set()
    out: list[str] = []
    for nt in md.notes:
        if nt.rest:
            out.append("R")
        elif nt.slur:
            out.append("-")
        elif not words:
            out.append("")
        else:
            k = _word_of(float(nt.start), ws, we)
            if k in shown:
                out.append("")
            else:
                shown.add(k)
                out.append(" ".join(_ph_disp(classifier, md.phones[i][2]) for i in groups[k]))
    return out


def note_labels(md: EditModel, classifier) -> list[str]:
    """音符标签入口：有 words 层 → 按 word 显示；翻新工程（无 words 层）→ 元音核心启发式。"""
    if md.data.words:
        return note_word_labels(md, classifier)
    return note_cv_labels(md, classifier)


def boundary_limits(md: EditModel, index: int,
                    end_limit: float | None = None) -> tuple[float, float]:
    """边界 index（1..n）的可移动范围 [lo, hi]：夹在相邻音素内部、互不穿越。"""
    p = md.phones
    n = len(p)
    lo = p[index - 1][0] + MIN_PHONE_DUR
    if index < n:
        hi = p[index][1] - MIN_PHONE_DUR
    else:
        hi = end_limit if end_limit is not None else float("inf")
    if hi < lo:
        hi = lo
    return lo, hi


def clamp_boundary(md: EditModel, index: int, t: float,
                   end_limit: float | None = None) -> float:
    lo, hi = boundary_limits(md, index, end_limit)
    return max(lo, min(hi, t))


# ---- 音素：公开命令 ------------------------------------------------------
def split_phone(model: EditModel, index: int, position: float) -> bool:
    """剪刀切分：点击位置一分为二，两半同名（连续模式语义）。"""
    if not (0 <= index < len(model.phones)):
        return False
    if not (model.phones[index][0] + 1e-3 < position < model.phones[index][1] - 1e-3):
        return False
    model.run("切分音素", lambda md: mut_split_phone(md, index, position))
    return True


def duplicate_phone(model: EditModel, index: int) -> bool:
    """Ctrl+D：中点切开、两半同名（连续模式语义）。"""
    if not (0 <= index < len(model.phones)):
        return False
    if model.phones[index][1] - model.phones[index][0] < 1e-3:
        return False
    model.run("复制音素", lambda md: mut_duplicate_phone(md, index))
    return True


def remove_phone(model: EditModel, index: int) -> bool:
    """Delete：删除音素；与前一音素连续时延长其 end（连续模式语义）。"""
    if not (0 <= index < len(model.phones)):
        return False
    model.run("删除音素", lambda md: mut_remove_phone(md, index))
    return True


def rename_phone(model: EditModel, index: int, name: str) -> bool:
    if not (0 <= index < len(model.phones)) or model.phones[index][2] == name:
        return False
    model.run("重命名音素", lambda md: mut_rename_phone(md, index, name))
    return True


# ======================================================================
# 音符：底层变更（mut_*）
# ======================================================================
def mut_split_note(md: EditModel, index: int, position: float) -> bool:
    if not (0 <= index < len(md.notes)):
        return False
    n0 = md.notes[index]
    if not (n0.start + 1e-3 < position < n0.end - 1e-3):
        return False
    old_end, midi, rest = n0.end, n0.midi, n0.rest
    n0.end = position
    md.notes.insert(index + 1, Note(position, old_end, midi, slur=True, rest=rest))
    return True


def mut_set_note_pitch(md: EditModel, index: int, midi: float) -> bool:
    """自由拖动改音高：**不限固定半音**，保留到 1 音分（音分级微调）。"""
    if not (0 <= index < len(md.notes)) or md.notes[index].rest:
        return False
    md.notes[index].midi = _clamp_pitch(round(float(midi), 2))
    return True


def mut_nudge_note_pitch(md: EditModel, index: int, cents: float) -> bool:
    if not (0 <= index < len(md.notes)) or md.notes[index].rest:
        return False
    md.notes[index].midi = _clamp_pitch(round(md.notes[index].midi + cents / 100.0, 2))
    return True


def _clamp_pitch(m: float) -> float:
    return max(PITCH_MIN, min(PITCH_MAX, m))


def nearest_phone_boundary(md: EditModel, t: float, tol: float) -> float | None:
    """容差内最近音素边界（供音符端点吸附跟随，约 20ms）。"""
    best, best_d = None, tol
    for (s, e, _m) in md.phones:
        for b in (s, e):
            d = abs(b - t)
            if d < best_d:
                best, best_d = b, d
    return best


def mut_move_note_edge(md: EditModel, index: int, which: str, t: float,
                       snap_tol: float = 0.02, phone_end_limit: float | None = None) -> bool:
    """拖动音符左/右边界：与相邻音符共享的边界一并移动 → **总时长不变**
    （守住 sum_note_dur == sum_ph_dur）；落点靠近音素边界则吸附（容差 snap_tol）。
    """
    notes = md.notes
    if not (0 <= index < len(notes)):
        return False
    if which == "start" and index == 0:
        return False          # 首音符左端固定，保证总时长
    if which == "end" and index == len(notes) - 1:
        return False          # 末音符右端固定
    nt = notes[index]

    if which == "start":
        lo = notes[index - 1].start + MIN_NOTE_DUR
        hi = nt.end - MIN_NOTE_DUR
    else:
        lo = nt.start + MIN_NOTE_DUR
        hi = (notes[index + 1].end - MIN_NOTE_DUR) if index + 1 < len(notes) else _song_end(md, phone_end_limit)

    snap = nearest_phone_boundary(md, t, snap_tol)
    if snap is not None and lo <= snap <= hi:
        t = snap
    t = max(lo, min(hi, t))
    if which == "start":
        nt.start = t
        notes[index - 1].end = t
    else:
        nt.end = t
        if index + 1 < len(notes):
            notes[index + 1].start = t
    md.notes.sort(key=lambda x: (x.start, x.end))
    return True


def _song_end(md: EditModel, phone_end_limit: float | None) -> float:
    if phone_end_limit is not None:
        return phone_end_limit
    if md.phones:
        return md.phones[-1][1]
    return float("inf")


def note_phone_labels(md: EditModel) -> list[str]:
    """音符块上显示的标签（音符上显示音素，切分多出的音符显示 -）。

    - 休止音符 → `R`
    - 切分产生的后半（slur）→ `-`（合成编辑器惯例）
    - 其余 → 该音符时间区间覆盖到的音素名（空格连接）
    """
    out: list[str] = []
    for nt in md.notes:
        if nt.rest:
            out.append("R")
            continue
        if nt.slur:
            out.append("-")
            continue
        names = [m or "∅" for (s, e, m) in md.phones
                 if e > nt.start + 1e-4 and s < nt.end - 1e-4]
        out.append(" ".join(names))
    return out


def brief_phone(classifier, ph: str) -> str:
    """音素显示名 = 本体（`zh/x` → `x`），与钢琴窗的显示宽度匹配。"""
    return classifier.split_phoneme(ph)[1] or ph


def note_label_indices(md: EditModel, classifier) -> list:
    """每个音符标签**对应的音素下标组**（休止音符 → `"R"`、连音音符 → `"-"`）。

    `note_cv_labels` 只是把这些下标转成音素名而已。算法按**元音核心**把音素序列划分给
    音节音符（**音素序列的一个划分**，因此天然不重复、不遗漏）：
      1. 先找每个**音节音符**（非休止 / 非连音）的**元音核心** = 与音符重叠时长最大的
         那个元音（不能取「区间里第一个元音」，相邻音节会因边界近似对齐而混进来）；
      2. 起音 `lo`：元音前紧邻的那个音素若是辅音/流音 → 借过来（只取前面的
         一个辅音），否则从元音本身开始；
      3. 结束 `hi`：到**下一个音节音符的起音前一个**音素为止 —— 于是「其他辅音」自然
         留在上一个音符、韵尾也留在本音符（「有其它辅音时留在上一个音符」
         「后面是无声音符时，结尾的辅音仍显示进当前音符」，两者都是本划分的特例）；
         末尾音符一直取到最后一个音素，但**结尾的无声音素（SP/AP…）不写进标签**。

    演进教训（别再退回去）：
    ① 曾把「前导辅音」限制成必须也在本音符区间内 → 声母 end 恰等于音符 start 时被判掉，
       中文一起坏（边界只是**近似**对齐）；
    ② 曾把「元音之后、与音符重叠的音素」**无条件**追加为韵尾 → 下一个音节的声母同时
       出现在前一个音符的韵尾和它自己音符的起音上，同一个辅音显示两次；
    ③ 元音核心曾取「区间里**第一个**元音」→ 相邻音节的元音因边界近似对齐而落进本音符
       区间，被当成"本音符的元音"：于是前一个辅音在本音符又显示一次，而本音节的音素
       全被 `shown` 吞掉（英文「有的两次、有的被吞」、中文「辅音显示两次」的真因）；
    ④ 曾按「音符时间区间」直接取覆盖音素（`note_phone_labels` 那种）→ 边界一错开，
       同一个音素会落进两个音符的区间，而真正的音素又被漏掉。
    流音**一直**是当辅音处理的（`classify` 返回 `liquid`，判定处 `("consonant","liquid")`
    并列），它不是病因。
    """
    ph = md.phones
    n = len(ph)

    def cat(i: int) -> str:
        return classifier.classify(ph[i][2])

    def overlap(i: int, nt) -> float:
        return max(0.0, min(ph[i][1], nt.end) - max(ph[i][0], nt.start))

    # ① 收集音节音符：(音符下标, 元音核心, 起音下标)
    syl: list[tuple[int, int | None, int]] = []
    for j, nt in enumerate(md.notes):
        if nt.rest or nt.slur:
            continue
        idx = [i for i in range(n)
               if ph[i][1] > nt.start + 1e-4 and ph[i][0] < nt.end - 1e-4]
        v = None
        if idx:
            cand = [i for i in idx if cat(i) == "vowel"] \
                or [i for i in idx if cat(i) == "global"]
            if cand:
                v = max(cand, key=lambda i: (overlap(i, nt), -i))
        if v is not None:
            lo = v - 1 if (v > 0 and cat(v - 1) in ("consonant", "liquid")) else v
        else:                                   # 纯辅音 / 纯静音音符：从它覆盖的首个音素起
            lo = idx[0] if idx else 0
        syl.append((j, v, lo))

    # ② 起音强制单调递增（相邻音节绝不抢同一个音素）
    prev = -1
    for m, (j, v, lo) in enumerate(syl):
        syl[m] = (j, v, max(lo, prev + 1, 0))
        prev = syl[m][2]

    # ③ 划分：本音节起音 → 下一音节起音的前一个音素
    groups: list[list[int]] = []
    for m, (_j, _v, lo) in enumerate(syl):
        hi = (syl[m + 1][2] - 1) if m + 1 < len(syl) else (n - 1)
        if hi < lo:
            hi = lo
        while hi > lo and cat(hi) == "global":
            hi -= 1
        groups.append(list(range(lo, min(hi, n - 1) + 1)) if n else [])

    out: list = []
    k = 0
    for nt in md.notes:
        if nt.rest:
            out.append("R")
        elif nt.slur:
            out.append("-")
        else:
            out.append(groups[k] if k < len(groups) else [])
            k += 1
    return out


def note_cv_labels(md: EditModel, classifier) -> list[str]:
    """音符块标签：`起音辅音 + 元音 + 本音符其余音素`（如 `x a` / `d ow n`）。

    按 `note_label_indices` 的**元音核心划分**显示，所以**同一个音素只显示一次**、
    也不会被吞掉。休止音符 → `R`；切分后半（slur）→ `-`；音素名只取本体（`zh/x` → `x`）。
    """
    return [" ".join(brief_phone(classifier, md.phones[i][2]) for i in g)
            if not isinstance(g, str) else g
            for g in note_label_indices(md, classifier)]


def mut_toggle_slur(md: EditModel, index: int) -> bool:
    if not (0 <= index < len(md.notes)) or md.notes[index].rest:
        return False
    md.notes[index].slur = not md.notes[index].slur
    return True


def mut_set_note_slur(md: EditModel, index: int, value: bool) -> bool:
    if not (0 <= index < len(md.notes)) or md.notes[index].rest:
        return False
    if md.notes[index].slur == bool(value):
        return False
    md.notes[index].slur = bool(value)
    return True


def mut_set_note_rest(md: EditModel, index: int, value: bool) -> bool:
    """设为 / 取消无声。

    取消无声时，休止音符的 midi 只是**占位**（GAME 与 csv 的 rest 一律写 0.0，
    低于 PITCH_MIN → 不属于可用音高）：借用时间上相邻的非休止音符音高，
    全是休止时用 DEFAULT_PITCH，否则音符会落在琴键区外看不见。
    """
    if not (0 <= index < len(md.notes)):
        return False
    n = md.notes[index]
    if n.rest == bool(value):
        return False
    n.rest = bool(value)
    if n.rest:
        n.midi, n.slur = 0.0, False         # 无声：midi 归零占位；无声音符不可能连音
    elif n.midi < PITCH_MIN:
        prev = next((md.notes[j] for j in range(index - 1, -1, -1)
                     if not md.notes[j].rest), None)
        nxt = next((md.notes[j] for j in range(index + 1, len(md.notes))
                    if not md.notes[j].rest), None)
        ref = prev or nxt
        n.midi = float(ref.midi) if ref is not None else DEFAULT_PITCH
    return True


def mut_merge_notes(md: EditModel, index: int) -> bool:
    if not (0 <= index < len(md.notes) - 1):
        return False
    a, b = md.notes[index], md.notes[index + 1]
    if abs(a.end - b.start) > 1e-3:
        return False
    a.end = b.end
    del md.notes[index + 1]
    return True


def mut_delete_note(md: EditModel, index: int) -> bool:
    if not (0 <= index < len(md.notes)):
        return False
    del md.notes[index]
    return True


# ---- 音符：公开命令 ------------------------------------------------------
def split_note(model: EditModel, index: int, position: float) -> bool:
    """双击音符块：在点击时间切分，后半标记为连音(slur)。"""
    if not (0 <= index < len(model.notes)):
        return False
    n0 = model.notes[index]
    if not (n0.start + 1e-3 < position < n0.end - 1e-3):
        return False
    model.run("切分音符", lambda md: mut_split_note(md, index, position))
    return True


def set_note_pitch(model: EditModel, index: int, midi: float) -> bool:
    if not (0 <= index < len(model.notes)):
        return False
    nt = model.notes[index]
    if nt.rest or abs(round(float(midi), 2) - nt.midi) < 1e-3:
        return False
    model.run("修改音高", lambda md: mut_set_note_pitch(md, index, midi))
    return True


def nudge_note_pitch(model: EditModel, index: int, cents: float) -> bool:
    """方向键微调：±100 音分 = 半音，±10 音分 = 音分级微调。"""
    if not (0 <= index < len(model.notes)) or model.notes[index].rest:
        return False
    model.run(f"微调音高 {cents:+.0f}音分",
              lambda md: mut_nudge_note_pitch(md, index, cents))
    return True


def toggle_slur(model: EditModel, index: int) -> bool:
    """切换连音标记：slur=True 表示与前一个音符连唱（同一词内）。"""
    if not (0 <= index < len(model.notes)) or model.notes[index].rest:
        return False
    model.run("切换连音", lambda md: mut_toggle_slur(md, index))
    return True


def set_note_slur(model: EditModel, index: int, value: bool) -> bool:
    """右键菜单「设为连音 / 取消连音」：按**目标状态**设置（比 toggle 更明确）。"""
    if not (0 <= index < len(model.notes)) or model.notes[index].rest:
        return False
    if model.notes[index].slur == bool(value):
        return False                        # 已是该状态 → 不记一条空操作
    model.run("设为连音" if value else "取消连音",
              lambda md: mut_set_note_slur(md, index, value))
    return True


def set_note_rest(model: EditModel, index: int, value: bool) -> bool:
    """右键菜单「设为无声 / 取消无声」。"""
    if not (0 <= index < len(model.notes)):
        return False
    if model.notes[index].rest == bool(value):
        return False
    model.run("设为无声" if value else "取消无声",
              lambda md: mut_set_note_rest(md, index, value))
    return True


def merge_notes(model: EditModel, index: int) -> bool:
    """合并连音：把相邻（前 end == 后 start）的两个音符合并。"""
    if not (0 <= index < len(model.notes) - 1):
        return False
    a, b = model.notes[index], model.notes[index + 1]
    if abs(a.end - b.start) > 1e-3:
        return False
    model.run("合并连音", lambda md: mut_merge_notes(md, index))
    return True


def delete_note(model: EditModel, index: int) -> bool:
    if not (0 <= index < len(model.notes)):
        return False
    model.run("删除音符", lambda md: mut_delete_note(md, index))
    return True
