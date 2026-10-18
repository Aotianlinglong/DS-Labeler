# -*- coding: utf-8 -*-
"""
命令模式撤销/重做。

- 每个编辑 = 一条 MutateCommand（记录 before/after 快照，fn 只执行一次）
- 快照基于「音素 + 音符」两份列表；单句量极小（<20s），快照式最稳
"""
from __future__ import annotations

import copy


class Command:
    label = "操作"

    def apply(self, model) -> None:      # 正向 + 重做
        raise NotImplementedError

    def revert(self, model) -> None:     # 撤销
        raise NotImplementedError


class MutateCommand(Command):
    """把一个修改函数包成可撤销命令。"""

    def __init__(self, label: str, fn):
        self.label = label
        self._fn = fn
        self._before = None
        self._after = None

    def apply(self, model) -> None:
        if self._after is None:
            self._before = model.snapshot()
            self._fn(model)
            self._after = model.snapshot()
        else:
            model.restore(self._after)
        model.dirty = True

    def revert(self, model) -> None:
        model.restore(self._before)
        model.dirty = True


class SnapshotCommand(Command):
    """由外部给出的 before/after 快照（用于把一段鼠标拖动合并成一条记录）。"""

    def __init__(self, label: str, before, after):
        self.label = label
        self._before = before
        self._after = after

    def apply(self, model) -> None:
        model.restore(self._after)
        model.dirty = True

    def revert(self, model) -> None:
        model.restore(self._before)
        model.dirty = True


class UndoStack:
    def __init__(self, limit: int = 300):
        self._undo: list[Command] = []
        self._redo: list[Command] = []
        self._limit = limit

    # ---- 查询 ---------------------------------------------------------
    def can_undo(self) -> bool:
        return bool(self._undo)

    def can_redo(self) -> bool:
        return bool(self._redo)

    def undo_label(self) -> str:
        return self._undo[-1].label if self._undo else ""

    def redo_label(self) -> str:
        return self._redo[-1].label if self._redo else ""

    # ---- 操作 ---------------------------------------------------------
    def push(self, cmd: Command, model) -> None:
        cmd.apply(model)
        self._undo.append(cmd)
        self._redo.clear()
        if len(self._undo) > self._limit:
            self._undo.pop(0)

    def undo(self, model) -> str | None:
        if not self._undo:
            return None
        cmd = self._undo.pop()
        cmd.revert(model)
        self._redo.append(cmd)
        return cmd.label

    def redo(self, model) -> str | None:
        if not self._redo:
            return None
        cmd = self._redo.pop()
        cmd.apply(model)
        self._undo.append(cmd)
        return cmd.label

    def clear(self) -> None:
        self._undo.clear()
        self._redo.clear()


def clone_notes(notes):
    return [copy.deepcopy(n) for n in notes]
