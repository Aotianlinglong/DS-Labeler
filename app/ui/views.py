# -*- coding: utf-8 -*-
"""
自绘视图（三面板共享同一时间轴）：

  两个可拖拽面板，共享同一 `ViewState`（时间视窗），左侧统一留白 `GUTTER`
  以保证时间轴严格对齐：
    - SpectroRollView ：**mel 频谱底图 + MIDI 钢琴卷帘叠加窗**（同窗、频谱互不分离，
                         靠「频谱透明度」拉杆叠加显示）：音素边界/标签 + 音符 + F0（最上层）
                         + 「音符波形」（贴在每个音符正下方，视图菜单可开关）
    - WaveformView    ：波形（按文件峰值自适应）+ 音素边界竖线 + 音素名标签带

编辑交互：
  左键拖音素边界（夹取不穿越、跨边界音符端点吸附跟随）、拖音符改音高
           （自由、不限半音）、方向键 ±1 半音 / Shift+方向键 ±10 音分微调
  双击音素 = 重命名；双击音符 = 切分（后半连音显示 -）
  中键：拖拽平移（内容跟手，三面板方向一致；钢琴窗可上下自由平移）
  滚轮映射（三面板语义一致）：滚轮=竖向滚动、Shift+滚轮=水平滚动、
           Ctrl+滚轮=时间轴缩放、Alt+滚轮=音高行高缩放（仅钢琴卷帘窗）
  右键菜单；Ctrl+Z 撤销、Ctrl+Y / Ctrl+Shift+Z 重做
"""
from __future__ import annotations

import math

import numpy as np
from PySide6.QtCore import Qt, QRectF, QPointF, Signal
from PySide6.QtGui import (
    QPainter, QColor, QPen, QFont, QFontMetricsF, QImage, QPolygonF, QLinearGradient,
    QBrush, QPainterPath,
)
from PySide6.QtWidgets import QWidget, QInputDialog, QMenu

from core import edit
from core.edit import EditModel
from core.notes import Note
from core.project import DEFAULT_COLORS
from core.spectrogram import SpecData, make_cmap, CMAPS, DEFAULT_CMAP

import ui.theme as th

GUTTER = 72          # 左侧统一留白：钢琴键列 / 刻度（两面板时间轴对齐）
BAND_H = 0           # 画布内音素标签带已移除（音素名只在顶部 TierBar 显示）
BOUND_TOL_PX = 6     # 边界命中容差（像素）
SCROLL_PX_PER_NOTCH = 36.0   # 钢琴窗滚轮一格 ≈ 36px（按行高换算成半音，视觉速度恒定）


def _rgb_of(value, fallback=(128, 128, 128)) -> tuple[int, int, int]:
    """把配色值（"#rrggbb" / "#rgb" / (r,g,b) / QColor）解析成 (r, g, b)。

    非法值回退 fallback（设置里读到的颜色坏掉时不会把界面画崩）。
    """
    if isinstance(value, QColor):
        return (value.red(), value.green(), value.blue())
    if isinstance(value, (tuple, list)) and len(value) >= 3:
        return (int(value[0]), int(value[1]), int(value[2]))
    s = str(value or "").strip().lstrip("#")
    if len(s) == 3:
        s = "".join(ch * 2 for ch in s)
    if len(s) >= 6:
        try:
            return (int(s[0:2], 16), int(s[2:4], 16), int(s[4:6], 16))
        except ValueError:
            pass
    return tuple(fallback)



class _TierRow:
    """一行层（texts / words / phones）：左侧层名 + 紧随其后的条目块序列。"""

    ROW_H = 26
    NAME_W = GUTTER   # 与画布 GUTTER 逐像素对齐（修复层信息与面板错位）

    def __init__(self, name: str, color: str, editable: bool = False):
        self.name = name
        self.color = QColor(color)
        self.editable = editable
        self.draggable = name in ("words", "texts")   # 边界可拖（层信息栏内编辑）
        self.items: list[tuple[float, float, str]] = []   # (start, end, mark)
        self.sel = -1

    def set_items(self, xmins, xmaxs, marks):
        self.items = [(float(a), float(b), str(m)) for a, b, m in zip(xmins, xmaxs, marks)]
        self.sel = -1

    def set_phones(self, phones):
        self.items = [(float(s), float(e), str(m)) for s, e, m in phones]


class TierBar(QWidget):
    """层信息条（条目带）：texts / words / phones 三行。

    每个条目是一个色块，宽度与该条目时长成正比（与下方画布同一时间轴），
    块内写标签文字，选中的块高亮。

    **words / texts 行可人工编辑**（phones 的编辑在下方画布）：
      - 左键拖条目边界（words 拖动时 texts 同时间对齐的边界一起移动；不动 phones）
      - 双击条目改名（phones 行的双击语义在画布里，不在此处理）
    `tier_names` 可裁剪显示的行（翻新工程只显示 phones 层时传 `("phones",)`），
    打开工程时可按工程类型用 `set_tier_names()` 换行集。
    """

    sig_phone_selected = Signal(int)
    sig_tier_selected = Signal(str, int)   # 选中 words / texts 层条目（tier, 下标）
    sig_viewport = Signal()          # 中键平移后通知其余面板同步重绘
    sig_changed = Signal()           # words / texts 编辑完成（视图据此重建 + 落撤销）

    _COLORS = {"texts": "#8fd0ff", "words": "#a6e3a1", "phones": "#ffd45a"}

    def __init__(self, state: "ViewState", parent=None,
                 tier_names=("texts", "words", "phones")):
        super().__init__(parent)
        self.state = state
        self.model = None                      # EditModel（words/texts 编辑需要）
        self._tier_names = tuple(tier_names)
        self.rows = [_TierRow(n, self._COLORS.get(n, "#ffd45a"), editable=True)
                     for n in self._tier_names]
        self._phone_row = next((i for i, r in enumerate(self.rows) if r.name == "phones"), -1)
        self.setFixedHeight(_TierRow.ROW_H * max(1, len(self.rows)))
        self.setMouseTracking(True)
        self._pan_anchor = None
        self._drag: dict | None = None

    def set_tier_names(self, names):
        """切换显示的行（标准工程 texts/words/phones；翻新工程只有 phones）。"""
        names = tuple(names)
        if names == self._tier_names:
            return
        self._tier_names = names
        self.rows = [_TierRow(n, self._COLORS.get(n, "#ffd45a"), editable=True)
                     for n in names]
        self._phone_row = next((i for i, r in enumerate(self.rows) if r.name == "phones"), -1)
        self.setFixedHeight(_TierRow.ROW_H * max(1, len(self.rows)))
        self._drag = None
        self.update()

    def set_model(self, model):
        """挂上当前句的编辑器（words / texts 编辑据此落撤销栈）。"""
        self.model = model
        self._drag = None
        self.state.sel_tier, self.state.sel_tier_idx = "", -1

    # ---- 数据 ----
    def _sync_sel(self) -> None:
        """把每行的高亮**完全对齐到 ViewState**（唯一事实源）。

        高亮只能有一个来源：`state.sel_tier/sel_tier_idx`（words / texts）与
        `state.sel_phone`（phones）。行内自己攒的 `row.sel` 一律由这里刷新，
        否则「先点 texts 再点 words」「画布上改选音素」这类切换会留下旧高亮。
        """
        st = self.state
        for r in self.rows:
            if r.name == "phones":
                r.sel = st.sel_phone if st.sel_phone >= 0 else -1
            else:
                r.sel = st.sel_tier_idx if st.sel_tier == r.name else -1

    def set_tier(self, name: str, xmins, xmaxs, marks):
        for r in self.rows:
            if r.name == name:
                r.set_items(xmins, xmaxs, marks)
                break
        if self.state.sel_tier == name and \
                not (0 <= self.state.sel_tier_idx < len(self._tier_items(name))):
            self.state.sel_tier, self.state.sel_tier_idx = "", -1   # 条目没了 → 选择失效
        self._sync_sel()
        self.update()

    def _tier_items(self, name: str):
        for r in self.rows:
            if r.name == name:
                return r.items
        return []

    def set_phones(self, phones):
        for r in self.rows:
            if r.name == "phones":
                r.set_phones(phones)
                break
        self._sync_sel()
        self.update()

    def set_sel_phone(self, idx: int):
        self.state.sel_phone = idx
        self._sync_sel()
        self.update()

    # ---- 坐标（与画布共用 state，但这里没有 GUTTER 留白，层名占左端） ----
    def _cw(self) -> float:
        return max(1, self.width() - _TierRow.NAME_W)

    def _x(self, t: float) -> float:
        return _TierRow.NAME_W + self.state.t_to_x(t, self._cw())

    def _t(self, x: float) -> float:
        return self.state.x_to_t(x - _TierRow.NAME_W, self._cw())

    def _sec_per_px(self) -> float:
        return self.state.span / self._cw()

    def _edge_near(self, row: "_TierRow", x: float) -> float | None:
        """离 x 最近的条目边时间（容差 BOUND_TOL_PX）；没有则 None。"""
        best, best_d = None, float(BOUND_TOL_PX)
        for s, e, _m in row.items:
            for t in (float(s), float(e)):
                d = abs(self._x(t) - x)
                if d < best_d:
                    best, best_d = t, d
        return best

    def _row_at(self, y: float) -> "_TierRow | None":
        ri = int(y // _TierRow.ROW_H)
        return self.rows[ri] if 0 <= ri < len(self.rows) else None

    def _item_at(self, row: "_TierRow", x: float) -> int:
        for i, (s, e, _m) in enumerate(row.items):
            if self._x(float(s)) <= x <= self._x(float(e)):
                return i
        return -1

    def _snap_lower(self, tier: str, t: float) -> float:
        """拖动 words / texts 边界时吸附到**下层已有边界**（容差 BOUND_TOL_PX）：
        words → phones 边界；texts → words + phones 边界。越近越优先，超出容差则原样。

        只影响落点，**不移动下层**（phones 永不被 words/texts 拖动带走）。
        """
        if self.model is None:
            return t
        refs = [("phones", self.model.phones)] if tier == "words" \
            else [("words", self.model.words), ("phones", self.model.phones)]
        best, best_d = t, self._sec_per_px() * BOUND_TOL_PX
        for _name, items in refs:
            for s, e, _m in items:
                for c in (float(s), float(e)):
                    d = abs(c - t)
                    if d < best_d:
                        best, best_d = c, d
        return best

    # ---- 绘制 ----
    def paintEvent(self, event):
        p = QPainter(self)
        p.fillRect(self.rect(), QColor("#15171c"))
        p.setFont(QFont("Microsoft YaHei", 8))
        for ri, row in enumerate(self.rows):
            y0 = ri * _TierRow.ROW_H
            # 条目块
            p.save()
            p.setClipRect(QRectF(_TierRow.NAME_W, y0, self.width() - _TierRow.NAME_W,
                                 _TierRow.ROW_H))
            for i, (s, e, m) in enumerate(row.items):
                x0, x1 = self._x(s), self._x(e)
                if x1 < _TierRow.NAME_W or x0 > self.width():
                    continue
                w = max(1.0, x1 - x0)
                sel = (row.editable and i == row.sel)
                fill = QColor(row.color)
                fill.setAlpha(120 if sel else 46)
                p.fillRect(QRectF(x0, y0 + 2, w, _TierRow.ROW_H - 4), fill)
                p.setPen(QPen(row.color, 2 if sel else 1))
                p.setBrush(Qt.NoBrush)
                p.drawRect(QRectF(x0, y0 + 2, w, _TierRow.ROW_H - 4))
                if w > 14 and m:
                    p.setPen(QColor("#e8ecf2") if sel else QColor("#aeb6c2"))
                    p.save()
                    p.setClipRect(QRectF(x0 + 1, y0, w - 2, _TierRow.ROW_H))
                    p.drawText(QRectF(x0 + 3, y0, w - 4, _TierRow.ROW_H),
                               Qt.AlignLeft | Qt.AlignVCenter, m)
                    p.restore()
            p.restore()
            # 层名（画在条目块之后，遮挡从左侧溢出的块边缘）
            p.fillRect(0, y0, _TierRow.NAME_W, _TierRow.ROW_H, QColor("#1b1d24"))
            p.setPen(row.color)
            p.drawText(QRectF(4, y0, _TierRow.NAME_W - 6, _TierRow.ROW_H),
                       Qt.AlignLeft | Qt.AlignVCenter, row.name)
            # 行底分隔线
            p.setPen(QPen(QColor("#262a33"), 1))
            p.drawLine(0, y0 + _TierRow.ROW_H - 1, self.width(), y0 + _TierRow.ROW_H - 1)
        # words / texts 行的条目边界线（可拖 → 给出"这里能拖"的视觉提示）
        for ri, row in enumerate(self.rows):
            if not row.draggable:
                continue
            y0, y1 = ri * _TierRow.ROW_H, ri * _TierRow.ROW_H + _TierRow.ROW_H - 1
            for s, e, _m in row.items:
                for t in (float(s), float(e)):
                    x = int(round(self._x(t)))
                    if x < _TierRow.NAME_W or x > self.width():
                        continue
                    p.setPen(QPen(QColor(row.color), 1))
                    p.drawLine(x, y0 + 1, x, y1)
        # phones 边界线：只贯穿 phones 行（texts/words 层不画边界线）
        if self._phone_row >= 0:
            prow = self.rows[self._phone_row]
            y_top = self._phone_row * _TierRow.ROW_H
            for s, e, _m in prow.items:
                for t in (s, e):
                    x = int(round(self._x(t)))
                    if x < _TierRow.NAME_W or x > self.width():
                        continue
                    p.setPen(QPen(QColor(0, 229, 255, 170), 1))
                    p.drawLine(x, y_top, x, self.height())
        # 拖动中的边界高亮
        if self._drag is not None:
            x = int(round(self._x(self._drag["cur_t"])))
            if x >= _TierRow.NAME_W:
                p.setPen(QPen(QColor(255, 255, 255, 210), 2))
                p.drawLine(x, 0, x, self.height())
        # 层名/内容分隔竖线
        p.setPen(QPen(QColor("#262a33"), 1))
        p.drawLine(_TierRow.NAME_W, 0, _TierRow.NAME_W, self.height())
        p.end()

    def mousePressEvent(self, event):
        if event.button() == Qt.MiddleButton:      # 中键水平平移（与其它面板同步）
            self._pan_anchor = (event.position().x(), self.state.t0)
            self.setCursor(Qt.ClosedHandCursor)
            return
        if event.button() != Qt.LeftButton:
            return
        pos = event.position()
        row = self._row_at(pos.y())
        if row is None:
            return
        # words / texts 行：靠近边界 → 起拖（拖动结束由 commit_snapshot 合并成一条撤销）
        if self.model is not None and row.draggable and pos.x() >= _TierRow.NAME_W:
            t = self._edge_near(row, pos.x())
            if t is not None and edit.tier_edges_at(row.items, t):
                self._drag = {"tier": row.name, "cur_t": t,
                              "before": self.model.snapshot()}
                self.setCursor(Qt.SizeHorCursor)
                self.update()
                return
        # 点选条目（phones 行沿用原「选中音素」语义；其余两行 = 选中该词/短语）
        i = self._item_at(row, pos.x())
        if i >= 0:
            if self._phone_row >= 0 and row is self.rows[self._phone_row]:
                # 选了音素 → words/texts 的选中作废（快捷键跟着最近的选择走）
                self.state.sel_tier, self.state.sel_tier_idx = "", -1
                self.state.sel_phone, self.state.sel_note = i, -1
                self._sync_sel()
                self.sig_phone_selected.emit(i)
            else:
                # 选了 words / texts 条目 → 音素 / 音符 / 另一行的选中一起作废
                self.state.sel_tier, self.state.sel_tier_idx = row.name, i
                self.state.sel_phone, self.state.sel_note = -1, -1
                self._sync_sel()
                self.sig_tier_selected.emit(row.name, i)
            self.update()

    def mouseMoveEvent(self, event):
        """中键水平平移；或在可编辑行上拖动边界（实时预览）。"""
        if self._pan_anchor is not None:
            x0, t0 = self._pan_anchor
            self.state.t0 = t0 - (event.position().x() - x0) / self._cw() * self.state.span
            self.state.clamp()
            self.update()
            self.sig_viewport.emit()
            return
        if self._drag is not None:
            raw = self._snap_lower(self._drag["tier"], self._t(event.position().x()))
            applied = edit.mut_move_tier_time(self.model, self._drag["tier"],
                                              self._drag["cur_t"], raw)
            if applied is not None:
                self._drag["cur_t"] = applied
            self.update()
            self.sig_viewport.emit()      # 与画布同步重绘
            return
        # 悬停在可拖边界上 → 左右调整光标
        row = self._row_at(event.position().y())
        if (self.model is not None and row is not None and row.draggable
                and event.position().x() >= _TierRow.NAME_W
                and self._edge_near(row, event.position().x()) is not None):
            self.setCursor(Qt.SizeHorCursor)
        else:
            self.unsetCursor()

    def mouseReleaseEvent(self, event):
        if self._pan_anchor is not None:
            self._pan_anchor = None
            self.unsetCursor()
            self.sig_viewport.emit()
            return
        if self._drag is not None:
            tier = self._drag["tier"]
            label = "移动词边界" if tier == "words" else "移动短语边界"
            changed = self.model.commit_snapshot(label, self._drag["before"])
            self._drag = None
            self.unsetCursor()
            self.update()
            self.sig_viewport.emit()
            if changed:
                self.sig_changed.emit()

    def mouseDoubleClickEvent(self, event):
        """words / texts 行双击条目改名（phones 行的双击语义在画布里，不在此处理）。"""
        if event.button() != Qt.LeftButton or self.model is None:
            return
        pos = event.position()
        row = self._row_at(pos.y())
        if row is None or not row.draggable:
            return
        i = self._item_at(row, pos.x())
        if i < 0:
            return
        cur = row.items[i][2]
        title = "重命名词" if row.name == "words" else "重命名短语"
        text, ok = QInputDialog.getText(self, title, f"{row.name} [{i + 1}] 名称：", text=cur)
        if not ok or text == cur:
            return
        if edit.rename_tier_item(self.model, row.name, i, text):
            self.update()
            self.sig_changed.emit()

    def wheelEvent(self, event):
        """与钢琴窗一致的滚轮语义：Shift+滚轮=水平滚动、Ctrl+滚轮=时间轴缩放；
        层信息条没有纵向轴 → 无修饰滚轮不动。"""
        ad = event.angleDelta()
        delta = ad.y() or ad.x()
        mods = event.modifiers()
        if mods & Qt.ControlModifier and not (mods & Qt.ShiftModifier):
            cw = max(1, self.width() - _TierRow.NAME_W)
            self.state.zoom(1.2 if delta > 0 else 1 / 1.2,
                            event.position().x() - _TierRow.NAME_W, cw)
        elif mods & Qt.ShiftModifier and not (mods & Qt.ControlModifier):
            self.state.scroll(-delta / 120.0 * self.state.span * 0.15)
        self.update()
        self.sig_viewport.emit()
        event.accept()


# ======================================================================
# 视窗状态（两面板共享；音素/音符选中态也放这里，顶部层信息栏读取）
# ======================================================================
class ViewState:
    def __init__(self, duration: float = 10.0):
        self.duration = max(0.01, duration)
        self.t0 = 0.0
        self.span = min(self.duration, 10.0)
        if self.span <= 0:
            self.span = 1.0
        # 选中（音素层）
        self.sel_phone = -1
        # 选中（音符层）
        self.sel_note = -1
        # 选中（words / texts 层条目）：只有在层信息栏里点中条目时才置上，
        # Delete / Ctrl+D / Ctrl+R 便作用于它而不是音素（避免快捷键混淆）；
        # 一旦在画布上选了音素/音符，这里立即清空（最近一次选择为准）。
        self.sel_tier = ""
        self.sel_tier_idx = -1

    def clamp(self):
        if self.span <= 0:
            self.span = 0.5
        self.span = min(self.span, self.duration)
        self.t0 = max(0.0, min(self.t0, max(0.0, self.duration - self.span)))

    def set_duration(self, duration: float):
        """切换句子：就地改时长（两面板共享同一对象，避免引用失效）。"""
        self.duration = max(0.01, float(duration))
        self.t0 = 0.0
        self.span = min(self.duration, 10.0)
        self.sel_phone = -1
        self.sel_note = -1
        self.sel_tier = ""
        self.sel_tier_idx = -1
        self.clamp()

    def x_to_t(self, x: float, cw: float) -> float:
        return self.t0 + x / max(1.0, cw) * self.span

    def t_to_x(self, t: float, cw: float) -> float:
        return (t - self.t0) / self.span * cw

    def fit(self):
        self.t0 = 0.0
        self.span = self.duration
        self.clamp()

    def zoom(self, factor: float, anchor_x: float, cw: float):
        t_anchor = self.x_to_t(anchor_x, cw)
        self.span = max(0.03, min(self.duration, self.span / factor))
        self.t0 = t_anchor - anchor_x / max(1.0, cw) * self.span
        self.clamp()

    def scroll(self, delta_sec: float):
        self.t0 += delta_sec
        self.clamp()


def _resample(a: np.ndarray, n_out: int, axis: int) -> np.ndarray:
    """把 a 沿 axis 重采样到 n_out：缩小取区间均值（消锯齿），放大取线性插值。"""
    n = a.shape[axis]
    if n_out <= 0:
        return a
    if n == n_out:
        return a.astype(np.float32)
    if n < n_out:
        # 线性插值：避免最近邻把 1 帧拉成整块像素 → 边缘呈阶梯状、像"错位"
        pos = np.linspace(0.0, n - 1.0, n_out)
        i0 = np.floor(pos).astype(np.int64)
        i1 = np.minimum(i0 + 1, n - 1)
        frac = (pos - i0).astype(np.float32)
        a0 = np.take(a.astype(np.float32), i0, axis=axis)
        a1 = np.take(a.astype(np.float32), i1, axis=axis)
        shape = [1] * a.ndim
        shape[axis] = n_out
        frac = frac.reshape(shape)
        return a0 * (1.0 - frac) + a1 * frac
    edges = np.unique(np.linspace(0, n, n_out + 1).astype(np.int32))
    sums = np.add.reduceat(a.astype(np.float32), edges[:-1], axis=axis)
    counts = np.diff(edges).astype(np.float32)
    shape = [1] * a.ndim
    shape[axis] = len(counts)
    return sums / counts.reshape(shape)


def _resample_crisp(a: np.ndarray, n_out: int, axis: int) -> np.ndarray:
    """缩小=均匀抽点（「点数/像素」语义：保谐波锐利，不平均发糊）；放大=线性插值。"""
    n = a.shape[axis]
    if n_out <= 0 or n == n_out:
        return a.astype(np.float32)
    if n > n_out:
        idx = np.round(np.linspace(0, n - 1, n_out)).astype(np.int64)
        return np.take(a, idx, axis=axis).astype(np.float32)
    return _resample(a, n_out, axis)


# ======================================================================
# 时间画布基类：统一坐标换算 / 刻度 / 播放头 / 平移 / 音素边界与标签带
# ======================================================================
class TimeCanvas(QWidget):
    sig_play_toggle = Signal()
    sig_play_phoneme = Signal(float, float)   # 右键播放所点音素（start, end）
    sig_changed = Signal()
    sig_status = Signal(str)
    sig_selection = Signal()
    sig_viewport = Signal()          # 视窗（t0/span/KEY_H）变化 → 主窗口同步刷新所有面板
    sig_cursor = Signal(float)       # 鼠标所在时间变化 → 主窗口广播，竖直虚线跨面板同步

    def __init__(self, state: ViewState, parent=None):
        super().__init__(parent)
        self.state = state
        self.model: EditModel | None = None
        self._play_pos = 0.0        # 播放头常驻可见（默认停在 0 点）
        self._cursor_t = -1.0
        self._pan_anchor = None
        self._samples: np.ndarray | None = None   # 当前句采样（波形面板 + 音符波形层共用）
        self._sr = 44100
        self._peak = 1.0
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.StrongFocus)
        self.setMinimumHeight(60)

    # ---- 坐标 ----
    def plot_w(self) -> int:
        return max(1, self.width() - GUTTER)

    def x_to_t(self, x: float) -> float:
        return self.state.x_to_t(x - GUTTER, self.plot_w())

    def t_to_x(self, t: float) -> float:
        return GUTTER + self.state.t_to_x(t, self.plot_w())

    def sec_per_px(self) -> float:
        return self.state.span / self.plot_w()

    def cursor_t(self) -> float:
        """鼠标所在时间（未在画布上时返回 -1）。"""
        return self._cursor_t

    def _set_cursor_t(self, t: float) -> None:
        """记录鼠标所在时间并发 `sig_cursor` 广播。

        光标竖直虚线必须**跨面板对齐**（否则每个面板各画各的，虚线不在同一竖线上）；
        故这里只负责记录 + 广播，实际重绘由主窗口统一分发给所有面板。
        """
        self._cursor_t = float(t)
        self.sig_cursor.emit(self._cursor_t)

    def _apply_cursor_t(self, t: float) -> None:
        """主窗口广播回来：更新本面板光标时间并重绘（**不再回发**，避免递归）。

        值未变则直接返回 —— 鼠标在 A 面板移动时，A 自身无需重绘两次。
        """
        t = float(t)
        if t == self._cursor_t:
            return
        self._cursor_t = t
        self.update()

    def leaveEvent(self, event):
        """鼠标离开画布 → 清掉竖直虚线（同样广播，免得其余面板残留一条线）。"""
        if self._cursor_t >= 0:
            self._set_cursor_t(-1.0)
        super().leaveEvent(event)

    def _clear_tier_sel(self) -> None:
        """画布上按下左键 → 层信息栏的 words / texts 选中作废（快捷键与高亮都跟着走）。

        会发 `sig_selection`，让主窗口刷新层信息栏（否则那一行的旧高亮会残留）。
        """
        if self.state.sel_tier:
            self.state.sel_tier, self.state.sel_tier_idx = "", -1
            self.sig_selection.emit()

    def set_play_pos(self, t: float):
        # 播放头常驻可见：负值一律夹回 0 点（切句传 -1 会导致指针"消失"）
        self._play_pos = max(0.0, float(t))
        self.update()

    def set_model(self, model: EditModel | None):
        self.model = model
        self._play_pos = 0.0
        self._cursor_t = -1.0
        self._pan_anchor = None
        self.update()

    def set_audio(self, samples: np.ndarray | None, sr: int):
        """挂上当前句音频（波形面板 + 钢琴卷帘的「音符波形」层共用同一份采样）。"""
        self._samples = samples
        self._sr = sr
        self._peak = (float(max(1e-6, np.max(np.abs(samples))))
                      if samples is not None and len(samples) else 1.0)
        self.update()

    # ---- 波形绘制（**独立波形面板与钢琴卷帘的波形段共用这一套**）----
    # 波形包络配色：**独立波形面板是蓝色**（这里默认），**钢琴卷帘侧改成浅灰白色**
    # （见 SpectroRollView 的覆盖）。两处共用同一套绘制算法与同一套时间映射，
    # 只有配色 + 可用竖直高度不同 —— 波形面板蓝、钢琴窗灰白。
    # 颜色可在设置→「配色」页自定义（或选配色方案），默认值统一取自 core.project.DEFAULT_COLORS。
    # 只填充、**不再画高亮描边**（描边会在放大后盖住波形细节，看不清）。
    WAVE_RGB_DEFAULT = _rgb_of(DEFAULT_COLORS["wave_panel_color"])   # 波形面板：蓝色
    WAVE_ALPHA = 255                        # 波形面板不透明
    WAVE_RGBA = (*WAVE_RGB_DEFAULT, WAVE_ALPHA)

    def set_wave_color(self, value=None) -> None:
        """设置波形包络颜色（"#rrggbb" / (r,g,b)）；alpha 按视图角色固定，不随自定义变。

        波形面板 = 不透明(255)，钢琴卷帘覆盖 WAVE_ALPHA=150 保留半透明观感。
        """
        if value is None:
            return
        r, g, b = _rgb_of(value, self.WAVE_RGB_DEFAULT)
        self.WAVE_RGBA = (r, g, b, self.WAVE_ALPHA)
        self.update()

    def _draw_wave_envelope(self, p: QPainter, rect: QRectF, fade_px: float = 0.0):
        """在 rect 内画波形包络 —— 独立波形面板（蓝）与钢琴卷帘波形段（浅灰白）共用。

        配色由 `WAVE_RGBA` 决定（钢琴卷帘侧覆盖成灰白），**只填充不描边**，
        **算法与时间映射两侧完全同一套**，所以形状/位置/大小/横向对齐不会两样。

        **映射铁律**：逐屏幕像素列用全局 `x_to_t` 反推该列的时间区间，放大档用全局
        `t_to_x` 逐点定位。**严禁**再用 `(t-t0)/(t1-t0)*rect.width()` 那种"拿 rect 宽度
        自己重算一遍"的局部线性映射 —— 钢琴卷帘侧的 xa/xb 是 floor/ceil 取整出来的，与
        `t_to_x(t0)/t_to_x(t1)` 并不严格相等，局部映射会随之漂移，会导致「上下波形根本错开」。
        rect 现在只负责**裁剪**与**竖直归一**（零线在 rect 竖直正中）。
        """
        smp = self._samples
        if smp is None or len(smp) == 0 or rect.width() < 2 or rect.height() < 8:
            return
        sr = float(self._sr)
        n = len(smp)
        mid = rect.center().y()
        amp = rect.height() * 0.47 / self._peak
        xa = max(GUTTER, int(math.floor(rect.left())))    # 只画到画布内（别画进左边键盘区）
        xb = min(self.width(), int(math.ceil(rect.right())))
        pwn = xb - xa
        if pwn < 2:
            return
        p.save()
        p.setClipRect(rect)
        p.setPen(QPen(QColor("#3a4049"), 1))               # 零线
        p.drawLine(QPointF(rect.left(), mid), QPointF(rect.right(), mid))
        fill = QColor(*self.WAVE_RGBA)
        if fade_px > 0 and rect.width() > 2 * fade_px:     # 段首尾 alpha 渐隐（「淡化」）
            fr = fade_px / rect.width()
            g = QLinearGradient(rect.left(), 0.0, rect.right(), 0.0)
            c0 = QColor(fill)
            c0.setAlpha(0)
            g.setColorAt(0.0, c0)
            g.setColorAt(fr, QColor(fill))
            g.setColorAt(1.0 - fr, QColor(fill))
            g.setColorAt(1.0, c0)
            brush, line_pen = QBrush(g), QPen(QBrush(g), 1)
        else:
            brush, line_pen = QBrush(fill), QPen(fill, 1)
        if self.sec_per_px() * sr >= 1.0:
            # 缩小档：每个屏幕像素列取该列时间区间内采样的 min/max，连成包络多边形
            ta, tb = self.x_to_t(xa), self.x_to_t(xb)
            idx = np.clip(np.round(np.linspace(ta, tb, pwn + 1) * sr).astype(np.int64), 0, n)
            lo = idx[:-1]
            hi = np.maximum(idx[1:], lo + 1)
            ys_t, ys_b = np.empty(pwn), np.empty(pwn)
            for k in range(pwn):
                a = smp[lo[k]:hi[k]]
                ys_t[k] = mid - (float(a.max()) if len(a) else 0.0) * amp
                ys_b[k] = mid - (float(a.min()) if len(a) else 0.0) * amp
            xs = np.arange(xa, xb, dtype=np.float64)
            # 必须用 WindingFill：包络多边形在安静段上/下两条边重合成一条零线（退化成零面积），
            # 默认的 OddEven 规则会让「安静段 + 某处一个尖峰」这种形状的尖峰 lobe 被判为
            # 偶数次穿越 → **不填充**（靠描边能遮掉这个 bug）。Winding 规则下正常填充。
            path = QPainterPath()
            path.setFillRule(Qt.WindingFill)
            path.addPolygon(QPolygonF(
                [QPointF(float(x), float(y)) for x, y in zip(xs, ys_t)]
                + [QPointF(float(x), float(y)) for x, y in zip(xs[::-1], ys_b[::-1])]))
            p.setPen(Qt.NoPen)                        # 只填充，不描边（描边会糊住波形细节）
            p.fillPath(path, brush)
        else:
            # 放大档：采样点比像素少时，按采样时间用 t_to_x 逐点定位连成折线（不挤在左边）
            ta = max(0.0, self.x_to_t(xa))
            tb = min(self.state.duration, self.x_to_t(xb))
            i0 = max(0, int(math.floor(ta * sr)))
            i1 = min(n, int(math.ceil(tb * sr)) + 1)
            if i1 > i0:
                p.setBrush(Qt.NoBrush)
                p.setPen(line_pen)
                p.drawPolyline(QPolygonF([
                    QPointF(float(self.t_to_x(i / sr)), float(mid - float(smp[i]) * amp))
                    for i in range(i0, i1)]))
        p.restore()

    # ---- 平移（中键 / 平移工具；三面板同一方向：内容跟手） ----
    def _start_pan(self, x: float, y: float = 0.0):
        self._pan_anchor = (x, y, self.state.t0)

    def _update_pan(self, x: float, y: float = 0.0) -> bool:
        if self._pan_anchor is None:
            return False
        x0, _y0, t0 = self._pan_anchor
        self.state.t0 = t0 - (x - x0) * self.sec_per_px()
        self.state.clamp()
        self.on_pan_vertical(y)      # 子类钩子：钢琴卷帘窗做垂直平移，其余面板只水平
        self.update()
        self.sig_viewport.emit()
        return True

    def on_pan_vertical(self, y: float) -> None:
        """中键自由平移的垂直分量。基类不做（波形/频谱只水平移动），钢琴卷帘窗覆写。"""

    def _end_pan(self) -> bool:
        if self._pan_anchor is None:
            return False
        self._pan_anchor = None
        self.setCursor(Qt.ArrowCursor)
        return True

    # ---- 公共绘制 ----
    def _paint_bg(self, p: QPainter, bg: str):
        p.fillRect(self.rect(), QColor(bg))
        p.fillRect(0, 0, GUTTER, self.height(), QColor("#1b1d24"))

    def _draw_play_cursor(self, p: QPainter):
        h = self.height()
        if self._cursor_t >= 0:
            x = self.t_to_x(self._cursor_t)
            p.setPen(QPen(QColor(120, 200, 255, 110), 1, Qt.DashLine))
            p.drawLine(int(x), 0, int(x), h)
        if self._play_pos >= 0:
            x = int(self.t_to_x(self._play_pos))
            p.setPen(QPen(QColor(255, 80, 80, 235), 2))
            p.drawLine(x, 0, x, h)
            # 顶部小三角手柄：让播放头在任何面板上都一眼可见
            tri = QPolygonF([QPointF(x - 5, 0), QPointF(x + 5, 0), QPointF(x, 8)])
            p.setBrush(QColor(255, 80, 80, 235))
            p.setPen(Qt.NoPen)
            p.drawPolygon(tri)
            p.setBrush(Qt.NoBrush)

    def _draw_frame(self, p: QPainter):
        # 必须先清掉画笔填充，否则会沿用上一次 setBrush 的颜色把整块面板填满
        p.setBrush(Qt.NoBrush)
        p.setPen(QPen(QColor(th.LINE), 1))
        p.drawRect(self.rect().adjusted(0, 0, -1, -1))

    def _draw_phone_boundaries(self, p: QPainter, phones, top: float, bottom: float):
        """贯穿竖线（TierBar→波形→频谱+MIDI 三面板逐像素对齐）；高亮青色双层加粗，
        确保在频谱亮区也清晰可辨。"""
        if not phones:
            return
        xs = [self.t_to_x(s) for s, _e, _m in phones]
        xs.append(self.t_to_x(phones[-1][1]))
        for x in xs:
            xi = int(round(x))
            p.setPen(QPen(QColor(0, 40, 48, 160), 3))            # 全高：深色衬底（亮区仍可辨）
            p.drawLine(xi - 1, int(top), xi - 1, int(bottom))
            p.setPen(QPen(QColor(0, 229, 255, 150), 1))          # 全高：亮青主线
            p.drawLine(xi, int(top), xi, int(bottom))
            p.setPen(QPen(QColor(0, 229, 255, 255), 1))          # 顶部刻度：实
            p.drawLine(xi - 1, int(top), xi - 1, int(top) + 10)
            p.drawLine(xi, int(top), xi, int(top) + 10)

    def _draw_phone_names(self, p: QPainter, phones, top: float):
        """面板顶部音素名，带语言标签：`zh/x` → 语言段 `zh` 用暗色、本体 `x` 用亮色
        （深色描边保证频谱亮区可读；无语言前缀的音素只画本体）。
        频谱+钢琴窗 / 口型面板共用这一套（与边界竖线同源，逐像素对齐）。"""
        cw = self.plot_w()
        p.setFont(QFont("Microsoft YaHei", 8))
        fm = p.fontMetrics()
        for (s, e, m) in phones:
            x0, x1 = self.t_to_x(s), self.t_to_x(e)
            if x1 < GUTTER or x0 > GUTTER + cw or x1 - x0 < 3:
                continue
            x0 = max(x0, GUTTER)          # 不盖到左侧琴键列
            mark = (m or "·")
            lang, _, body = mark.partition("/")
            text = mark if body else lang  # 无 "/" 时整体都是本体
            rect = QRectF(x0 + 2, top + 1, max(1.0, x1 - x0 - 3), 15)
            p.save()
            p.setClipRect(rect.adjusted(-2, -1, 0, 1))
            # 深色描边（频谱亮区也可读）
            p.setPen(QColor(0, 0, 0, 210))
            p.drawText(rect.adjusted(1, 1, 1, 1), Qt.AlignLeft | Qt.AlignVCenter, text)
            if body:                       # 语言标签（暗色）+ 本体（亮色）
                lp = QRectF(rect.x(), rect.y(), fm.horizontalAdvance(lang + "/"), rect.height())
                p.setPen(QColor("#9aa3ad"))
                p.drawText(lp, Qt.AlignLeft | Qt.AlignVCenter, lang + "/")
                bp = QRectF(rect.x() + lp.width(), rect.y(),
                            max(1.0, rect.width() - lp.width()), rect.height())
                p.setPen(QColor("#eef2f8"))
                p.drawText(bp, Qt.AlignLeft | Qt.AlignVCenter, body)
            else:
                p.setPen(QColor("#eef2f8"))
                p.drawText(rect, Qt.AlignLeft | Qt.AlignVCenter, text)
            p.restore()

    # ---- 缩放/滚动（视窗变化 → 同步刷新所有面板，避免"各缩各的"割裂感） ----
    def wheelEvent(self, event):
        """操作映射（三面板语义一致）：

          滚轮             —— 竖向滚动（钢琴窗=上下滚音高；波形/层信息条无纵向轴 → 不动）
          Shift+滚轮       —— 水平滚动（时间轴）
          Ctrl+滚轮        —— 时间轴缩放（以鼠标 x 为锚点）
          Alt+滚轮         —— 音高行高缩放（仅钢琴卷帘窗有纵向轴）
          Ctrl+Shift+滚轮  —— 同上（兼容别名）
        """
        ad = event.angleDelta()
        delta = ad.y() or ad.x()      # 兼容把滚轮上报成水平滚动的驱动（否则手势看似无效）
        mods = event.modifiers()
        if mods & Qt.ControlModifier:
            if mods & Qt.ShiftModifier:
                self.on_row_height_wheel(delta, event.position())
            else:
                self._on_ctrl_wheel(delta, event.position())
        elif mods & Qt.ShiftModifier:
            self.state.scroll(-delta / 120.0 * self.state.span * 0.15)
        elif mods & Qt.AltModifier:
            self.on_row_height_wheel(delta, event.position())
        else:
            self.on_plain_wheel(delta)
        self.update()
        self.sig_viewport.emit()
        event.accept()

    def _on_ctrl_wheel(self, delta: float, pos):
        """Ctrl+滚轮 = 时间轴缩放（以鼠标 x 为锚点）。"""
        self.state.zoom(1.2 if delta > 0 else 1 / 1.2,
                        pos.x() - GUTTER, self.plot_w())

    def on_row_height_wheel(self, delta: float, pos) -> None:
        """Alt+滚轮 = 音高行高缩放（只有钢琴卷帘窗有纵向轴，基类不做）。"""

    def on_plain_wheel(self, delta: float) -> None:
        """滚轮 = 竖向滚动。波形面板没有纵向轴 → 不动（与钢琴窗语义保持一致）。"""

    # ---- 音素边界拖动（两面板共用） ----
    def _try_boundary_drag(self, t: float) -> dict | None:
        if self.model is None:
            return None
        tol = self.sec_per_px() * BOUND_TOL_PX
        bi = self.model.boundary_near(t, tol)
        if bi is None:
            return None
        return {"kind": "boundary", "index": bi,
                "before": self.model.snapshot()}

    def _apply_boundary(self, drag: dict, t: float):
        self.model_apply_boundary(drag["index"], t)

    def model_apply_boundary(self, index: int, t: float):
        edit.mut_set_boundary(self.model, index, t, end_limit=self.state.duration)


# ======================================================================
# 波形（按文件峰值自适应）+ 音素边界竖线 + 音素名
# ======================================================================
class WaveformView(TimeCanvas):
    def __init__(self, state: ViewState, parent=None):
        super().__init__(state, parent)
        self._drag: dict | None = None

    def paintEvent(self, event):
        p = QPainter(self)
        self._paint_bg(p, th.WAVE_BG)
        w, h = self.width(), self.height()
        top, bottom = BAND_H, h
        phones = self.model.phones if self.model else []

        if self._samples is None or len(self._samples) == 0:
            p.setPen(QColor("#667"))
            p.drawText(QRectF(GUTTER, top, self.plot_w(), max(1, bottom - top)),
                       Qt.AlignCenter, "无音频")
            self._draw_phone_boundaries(p, phones, top, bottom)
            self._draw_play_cursor(p)
            self._draw_frame(p)
            p.end()
            return

        # 波形本体交给基类的共用实现（与钢琴卷帘的波形段**逐像素同一套算法**，
        # 所以两处波形形状/配色/振幅比例完全一致，只是竖直空间大小不同）
        self._draw_wave_envelope(p, QRectF(GUTTER, top, self.plot_w(), max(1, bottom - top)))
        # 音素边界竖线贯穿波形（与 TierBar / 频谱+钢琴窗逐像素对齐）
        self._draw_phone_boundaries(p, phones, top, bottom)

        # 左侧名称提示
        p.setPen(QColor("#6b7280"))
        p.setFont(QFont("Consolas", 8))
        p.drawText(QRectF(4, top, GUTTER - 8, 16), Qt.AlignLeft | Qt.AlignVCenter, "波形")
        self._draw_play_cursor(p)
        self._draw_frame(p)
        p.end()

    # ---- 交互（中键平移方向与其余面板一致） ----
    def mousePressEvent(self, event):
        pos = event.position()
        btn = event.button()
        t = self.x_to_t(pos.x())
        self._set_cursor_t(t if pos.x() >= GUTTER else -1.0)
        if btn == Qt.MiddleButton:                 # 中键平移（与其它面板同步）
            self._start_pan(pos.x(), pos.y())
            self.setCursor(Qt.ClosedHandCursor)
            return
        # 左键：波形面板同样可拖音素边界（与频谱+MIDI 面板行为一致、双向同步）
        if btn == Qt.LeftButton and pos.x() >= GUTTER and self.model is not None:
            self._clear_tier_sel()      # 层信息栏的词/短语高亮一并作废
            drag = self._try_boundary_drag(t)
            if drag is not None:
                self._drag = drag
                self.setCursor(Qt.SizeHorCursor)
                return
            idx = self.model.phone_index_at(t)
            self.state.sel_phone, self.state.sel_note = idx, -1
            self.sig_selection.emit()
        self.update()

    def mouseMoveEvent(self, event):
        pos = event.position()
        self._set_cursor_t(self.x_to_t(pos.x()) if pos.x() >= GUTTER else -1.0)
        if self._update_pan(pos.x(), pos.y()):
            return
        if self._drag is not None:
            self._apply_boundary(self._drag, self.x_to_t(pos.x()))
            self.update()
            self.sig_viewport.emit()          # 与频谱+MIDI 面板同步重绘
            return
        self.update()

    def mouseReleaseEvent(self, event):
        if self._end_pan():
            return
        if self._drag is not None:
            changed = self.model.commit_snapshot("移动音素边界", self._drag["before"])
            self._drag = None
            self.setCursor(Qt.ArrowCursor)
            self.update()
            self.sig_viewport.emit()
            if changed:
                self.sig_changed.emit()
            return
        self.update()


# ======================================================================
# 口型面板（R3MOE 开口度曲线）：填充 + 描边，两色可自定义；背景用软件主题色
# ======================================================================
class CurveCanvas(TimeCanvas):
    """口型（开口度）曲线面板：**只读**显示，交互与其余面板一致（平移/缩放/选中音素）。

    - 曲线取自 R3MOE（core/curve.py），值 = 归一化开口度（1 = 最大张口）；
    - 纵向标尺：默认**按当前句峰值自适应**（把该句最大张口顶到面板顶端，形状才看得清），
      可在设置→「面板」页或「显示」菜单关掉 → 退回绝对标尺（0 底 / 1 顶）；
    - 画法：曲线下方**填充** + 曲线**描边**（两色来自设置→「配色」页，可自定义）；
    - 顶部画音素边界竖线与音素名（与钢琴卷帘同源、逐像素对齐）；
    - 背景刻意用软件主题色（`th.CANVAS_BG`），**不用 #2c2c2c 那类深灰**；
    - 曲线不可编辑，只做浏览。
    """

    LINE_RGB_DEFAULT = _rgb_of(DEFAULT_COLORS["curve_line_color"])   # 描边默认色
    FILL_RGB_DEFAULT = _rgb_of(DEFAULT_COLORS["curve_fill_color"])   # 填充默认色

    def __init__(self, state: ViewState, parent=None):
        super().__init__(state, parent)
        self._curve: np.ndarray | None = None
        self._curve_fps = 50.0
        self._autoscale = True                 # 纵轴按当前句峰值自适应（默认开）
        self._drag: dict | None = None
        self._line = QColor(*self.LINE_RGB_DEFAULT)
        self._fill = QColor(*self.FILL_RGB_DEFAULT)

    # ---- 数据 / 配色 ----
    def set_curve(self, curve: np.ndarray | None, fps: float = 50.0):
        self._curve = (np.asarray(curve, dtype=np.float32) if curve is not None
                       and len(curve) else None)
        self._curve_fps = float(fps) if fps and fps > 0 else 50.0
        self.update()

    def set_autoscale(self, on: bool) -> None:
        """纵轴是否按当前句峰值自适应（软件级设置，设置→「面板」页 / 「显示」菜单同源）。"""
        self._autoscale = bool(on)
        self.update()

    def set_curve_colors(self, line=None, fill=None) -> None:
        """设置口型曲线的描边 / 填充色（"#rrggbb" / (r,g,b)）；不改则保留原色。"""
        if line is not None:
            self._line = QColor(*_rgb_of(line, self.LINE_RGB_DEFAULT))
        if fill is not None:
            self._fill = QColor(*_rgb_of(fill, self.FILL_RGB_DEFAULT))
        self.update()

    def _y_of(self, v: float, y0: float, span: float) -> float:
        """开口度 → 屏幕 y：0 = 基线(闭嘴，靠下)，1 = 顶端(最大张口)；负值夹到基线。"""
        return y0 - float(min(1.0, max(0.0, v))) * span

    def _vscale(self) -> float:
        """纵轴满量程对应的开口度：自适应 = 当前句峰值（下限 0.05，避免静音句被放大成噪声），
        否则 = 1.0（绝对标尺）。"""
        if not self._autoscale:
            return 1.0
        if self._curve is None or len(self._curve) == 0:
            return 1.0
        return max(0.05, min(1.0, float(self._curve.max())))

    def peak(self) -> float:
        """当前句开口度峰值（1 = 最大张口）；无数据时 0。"""
        return (float(self._curve.max())
                if self._curve is not None and len(self._curve) else 0.0)

    # ---- 绘制 ----
    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing, True)   # 曲线 / 边界线抗锯齿（默认关，开启更平滑）
        self._paint_bg(p, th.CANVAS_BG)
        w, h = self.width(), self.height()
        top, bottom = BAND_H, h
        phones = self.model.phones if self.model else []

        if self._curve is None or len(self._curve) == 0:
            p.setPen(QColor("#667"))
            p.drawText(QRectF(GUTTER, top, self.plot_w(), max(1, bottom - top)),
                       Qt.AlignCenter, "无口型数据")
        else:
            self._draw_curve(p, QRectF(GUTTER, top, self.plot_w(), max(1, bottom - top)))
        self._draw_phone_boundaries(p, phones, top, bottom)
        # 音素名（与边界竖线同源；只画名字不画音素块，观感与钢琴卷帘一致）
        self._draw_phone_names(p, phones, top)

        # 左侧名称提示 + （自适应时）当前句峰值：让用户知道纵轴不是绝对 0…1
        p.setPen(QColor("#6b7280"))
        p.setFont(QFont("Consolas", 8))
        p.drawText(QRectF(4, top, GUTTER - 8, 16), Qt.AlignLeft | Qt.AlignVCenter, "口型")
        if self._curve is not None and len(self._curve):
            p.setFont(QFont("Consolas", 7))
            p.drawText(QRectF(4, top + 15, GUTTER - 8, 13), Qt.AlignLeft | Qt.AlignVCenter,
                       f"峰 {self.peak():.2f}")
        self._draw_play_cursor(p)
        self._draw_frame(p)
        p.end()

    def _draw_curve(self, p: QPainter, rect: QRectF):
        """在 rect 内画口型曲线（填充 + 描边）。时间映射走全局 x_to_t / t_to_x，
        与其余面板逐像素对齐（严禁用 rect 宽度自己重算局部线性映射）。

        纵向：`_vscale()` 给满量程（自适应=当前句峰值 / 绝对=1.0），所有值先除以它
        再映射到基线→顶端，于是自适应下该句峰值恒顶到面板顶部。
        """
        cv = self._curve
        fps = self._curve_fps
        inv = 1.0 / self._vscale()          # 归一到「满量程」
        pad = min(8.0, rect.height() * 0.06)
        y0 = rect.bottom() - pad            # 闭嘴基线
        span = max(1.0, rect.height() - 2 * pad)
        xa = max(GUTTER, int(math.floor(rect.left())))
        xb = min(self.width(), int(math.ceil(rect.right())))
        pwn = xb - xa
        if pwn < 2:
            return
        xs: list[float] = []
        ys: list[float] = []
        if self.sec_per_px() * fps >= 1.0:
            # 缩小档：每屏幕像素列取该列时间区间内曲线的**峰值**（张口更醒目）
            ta, tb = self.x_to_t(xa), self.x_to_t(xb)
            idx = np.clip(np.linspace(ta * fps, tb * fps, pwn + 1).astype(np.int64),
                          0, len(cv))
            for k in range(pwn):
                lo, hi = int(idx[k]), max(int(idx[k + 1]), int(idx[k]) + 1)
                seg = cv[lo:hi]
                v = float(seg.max()) if len(seg) else 0.0
                xs.append(float(xa + k))
                ys.append(self._y_of(v * inv, y0, span))
        else:
            # 放大档：逐帧用 t_to_x 定位（线性插值交给 Qt 折线，不挤在左边）
            ta = max(0.0, self.x_to_t(xa))
            tb = min(self.state.duration, self.x_to_t(xb))
            i0 = max(0, int(math.floor(ta * fps)))
            i1 = min(len(cv), int(math.ceil(tb * fps)) + 1)
            for i in range(i0, i1):
                xs.append(float(self.t_to_x(i / fps)))
                ys.append(self._y_of(float(cv[i]) * inv, y0, span))
        if len(xs) < 2:
            return
        p.save()
        p.setClipRect(rect)
        p.setPen(QPen(QColor("#3a4049"), 1))              # 闭嘴基线
        p.drawLine(QPointF(rect.left(), y0), QPointF(rect.right(), y0))
        # 填充：曲线 + 基线围成的区域
        path = QPainterPath()
        path.setFillRule(Qt.WindingFill)
        path.addPolygon(QPolygonF(
            [QPointF(x, y) for x, y in zip(xs, ys)]
            + [QPointF(xs[-1], y0), QPointF(xs[0], y0)]))
        p.setPen(Qt.NoPen)
        p.fillPath(path, QBrush(self._fill))
        # 描边：沿曲线画折线
        p.setBrush(Qt.NoBrush)
        p.setPen(QPen(self._line, 1.6))
        p.drawPolyline(QPolygonF([QPointF(x, y) for x, y in zip(xs, ys)]))
        p.restore()

    # ---- 交互（曲线本身不可编辑；其余与钢琴卷帘一致：中键平移 / 右键播放所点音素 /
    #      左键选中音素与拖边界）----
    def mousePressEvent(self, event):
        pos = event.position()
        btn = event.button()
        t = self.x_to_t(pos.x())
        self._set_cursor_t(t if pos.x() >= GUTTER else -1.0)
        if btn == Qt.MiddleButton:
            self._start_pan(pos.x(), pos.y())
            self.setCursor(Qt.ClosedHandCursor)
            return
        # 右键：播放所点音素（与钢琴卷帘同一手势；本面板只读，故不给音符菜单）
        if btn == Qt.RightButton and pos.x() >= GUTTER and self.model is not None:
            idx = self.model.phone_index_at(t)
            if idx >= 0:
                s, e, _m = self.model.phones[idx]
                self.state.sel_phone, self.state.sel_note = idx, -1
                self.sig_selection.emit()
                self.sig_play_phoneme.emit(s, e)
                self.update()
            return
        if btn == Qt.LeftButton and pos.x() >= GUTTER and self.model is not None:
            self._clear_tier_sel()
            drag = self._try_boundary_drag(t)
            if drag is not None:
                self._drag = drag
                self.setCursor(Qt.SizeHorCursor)
                return
            idx = self.model.phone_index_at(t)
            self.state.sel_phone, self.state.sel_note = idx, -1
            self.sig_selection.emit()
        self.update()

    def mouseMoveEvent(self, event):
        pos = event.position()
        self._set_cursor_t(self.x_to_t(pos.x()) if pos.x() >= GUTTER else -1.0)
        if self._update_pan(pos.x(), pos.y()):
            return
        if self._drag is not None:
            self._apply_boundary(self._drag, self.x_to_t(pos.x()))
            self.update()
            self.sig_viewport.emit()
            return
        self.update()

    def mouseReleaseEvent(self, event):
        if self._end_pan():
            return
        if self._drag is not None:
            changed = self.model.commit_snapshot("移动音素边界", self._drag["before"])
            self._drag = None
            self.setCursor(Qt.ArrowCursor)
            self.update()
            self.sig_viewport.emit()
            if changed:
                self.sig_changed.emit()
            return
        self.update()


# ======================================================================
# mel 频谱 + MIDI 钢琴卷帘 叠加窗（同窗叠加：频谱透明度可调）
# ======================================================================
def word_start_notes(notes) -> list[int]:
    """在块**左上角外侧**额外写一份标签的音符下标 = **除休止外的所有音符**。

    休止音符不写（块内根本不画文字）；**连音音符要写**，它的标签就是 `-`
    （连音符上显示 -）—— 块内文字默认关之后，`-` 只能靠块外这份标签才看得见。
    写在块**外侧上方**的原因：PIT 曲线穿过音符块时会盖住块内文字，看不清。
    """
    return [i for i, n in enumerate(notes) if not n.rest]


CORNER_LABEL_PAD = 4.0   # 词首标签基线相对块顶再留的净空（px）：调大 = 整行标签更往上


def corner_label_pos(rect_top: float, fm, x: float) -> QPointF:
    """词首音符标签的落笔点（QPainter.drawText(QPointF, str) 用的是**基线**）。

    基线取 `块顶 − descent − CORNER_LABEL_PAD`：整字（含 g / y 的下伸部）都落在
    音符块**上方外侧**，且再留一点净空，不贴块顶。
    """
    return QPointF(x, rect_top - fm.descent() - CORNER_LABEL_PAD)


class SpectroRollView(TimeCanvas):
    """频谱 / 钢琴卷帘画布。`layers` 选择画哪几层，用于把两窗分开：

      layers=("notes",)       → 只做钢琴卷帘（音高网格 + 音符 + F0 + 琴键列）
      layers=("spec",)        → 只做频谱窗（mel 底图，无音高纵轴、不可拖音符）
      layers=("spec","notes") → 旧的一体化叠加窗（频谱底图 + MIDI，透明度联动）

    两窗共用同一个 `ViewState`（同一套浮点时间映射 t_to_x），拆开后时间轴仍逐像素
    对齐；音素边界杠与音素名两窗都画（对齐判断的公共参照）。频谱窗里不响应音符编辑
    （音符在频谱窗上不可见，若仍可拖会变成「看不见的误改」）。
    """

    KEY_H = 30.0                 # 音高行高（默认黑键≈30px 的观感；Alt+滚轮缩放）
    LABEL_PT = 7.0               # 音符块内标签 / 词首标签 / 音素名的字号（软件级设置可改）
    MERGE_TOL_PX = 14.0          # 双击合并：共享边界前后各 ±14px 内都算「断裂命中」（5px 太窄）
    # 「波形预览」：音符行下方一条**完整的**波形（与独立波形面板同一套绘制，
    # 一组一组画、不与音符块相贴、段首尾渐隐）
    WAVE_GAP = 38.0              # 波形带顶部相对该组最低音行下沿的空隙（固定像素）
    WAVE_H_DEFAULT = 42.0        # 波形包络上下最大宽度（像素；纵向不随 KEY_H 缩放，设置面板可改）
    WAVE_EXT_MS_DEFAULT = 80.0   # 「错开时多显示一截」的外扩**时长**（毫秒；设置面板可改）
    WAVE_FADE_PX = 16.0          # 段首尾的透明渐隐宽度（视觉渐隐：错开的地方有淡化）
    # 一段波形允许的**累计音高跨度**上限（半音）：段内最高音 − 最低音 超过它就另起一段。
    # 判据必须用累计跨度而**不是**相邻两音的距离——后者会让一路级进爬升/下降
    # （每步都小）永远串成一段，波形段跟不上旋律线条。
    # 阈值取 3 半音（与原相邻差判据同量级）。
    WAVE_GROUP_MAX_SPAN = 3.0
    WAVE_NAME_RGBA = (233, 240, 250, 210)   # 波形带上的音素名（宽字距浅白）
    PIT_W_DEFAULT = 1.5          # PIT（F0）音高线粗细（px，设置面板可调：1.0 最细 / 2.0 最粗）
    PIT_ALPHA = 205              # PIT 线 alpha（固定：自定义只改色相不改通透度）
    # 钢琴卷帘侧的波形配色（默认取自 core.project.DEFAULT_COLORS 的深靛蓝）。
    # 刻意**不用中性灰白**——灰白与常见音频工具观感撞脸，辨识度低。
    WAVE_RGB_DEFAULT = _rgb_of(DEFAULT_COLORS["wave_roll_color"])
    WAVE_ALPHA = 150                                      # 半透明（与波形面板不透明区分角色）
    WAVE_RGBA = (*WAVE_RGB_DEFAULT, WAVE_ALPHA)
    # 普通音符配色：填充 alpha 固定，描边由填充色向白混合得到（默认蓝 → 接近原来的
    # (235,250,255)）。休止(rest)/连音(slur)/选中框是**语义标记色**，保持固定不参与自定义。
    NOTE_ALPHA = 205
    NOTE_BORDER_ALPHA = 220
    NOTE_WHITE_MIX = 0.85        # 描边 = 填充色向白混合这个比例
    KEYS = range(12, 120)

    @property
    def WAVE_H(self) -> float:
        """波形包络上下最大宽度（像素）——纵向不随缩放变，设置面板可自定义。"""
        return self._wave_h

    @property
    def WAVE_EXT_S(self) -> float:
        """每段波形首尾的横向外扩**时长**（秒）——设置面板可自定义。

        必须用**固定时长**而不是固定像素：像素外扩会随时间轴缩放变长变短（缩得越小，
        同样多显示一点就要占越多像素），那就不是稳定的「多显示一截」了。
        """
        return self._wave_ext_s

    def set_wave_metrics(self, height: float | None = None,
                         ext_ms: float | None = None) -> None:
        """按工程设置自定义钢琴窗波形的上下最大宽度（px）与外扩时长（ms）。"""
        if height is not None:
            self._wave_h = max(8.0, min(400.0, float(height)))
        if ext_ms is not None:
            self._wave_ext_s = max(0.0, min(5.0, float(ext_ms) / 1000.0))
        self.update()

    def set_pit_width(self, w: float | None) -> None:
        """PIT（F0）音高线的粗细（px）：1.0 = 最细（修订后那版）、2.0 = 更早那版最粗。"""
        if w is not None:
            self._pit_w = max(1.0, min(2.0, float(w)))
        self.update()

    def set_label_font(self, pt: float | None) -> None:
        """标签字号（pt）：音符块内标签 / 词首标签 / 波形带上的音素名共用一套。

        **软件级**设置（设置 →「面板」页「标签字号」），不是工程级。范围 5–16pt：
        再小看不清、再大会盖住相邻行。
        """
        if pt is not None:
            self._label_pt = max(5.0, min(16.0, float(pt)))
        self.update()

    # ---- 配色（钢琴卷帘用到的三种：波形 / 普通音符 / PIT）-------------------
    def set_note_colors(self, value=None) -> None:
        """设置**普通音符**的填充色（"#rrggbb" / (r,g,b)）。

        描边由填充色向白混合得到（保证自定义色下仍有清晰轮廓）；休止(rest)、
        连音(slur)、选中高亮框都是**语义标记色**，保持固定、不随自定义变，避免失去区分度。
        """
        if value is None:
            return
        r, g, b = _rgb_of(value, self._note_rgb)
        self._note_rgb = (r, g, b)
        self._note_fill = QColor(r, g, b, self.NOTE_ALPHA)
        m = self.NOTE_WHITE_MIX
        self._note_border = QColor(int(r + (255 - r) * m), int(g + (255 - g) * m),
                                   int(b + (255 - b) * m), self.NOTE_BORDER_ALPHA)
        self.update()

    def set_pit_color(self, value=None) -> None:
        """设置 PIT（F0）音高线的颜色（"#rrggbb" / (r,g,b)）；alpha 固定 = PIT_ALPHA。"""
        if value is None:
            return
        r, g, b = _rgb_of(value, self._pit_rgb)
        self._pit_rgb = (r, g, b)
        self._pit_color = QColor(r, g, b, self.PIT_ALPHA)
        self.update()

    def __init__(self, state: ViewState, parent=None, layers=("spec", "notes")):
        super().__init__(state, parent)
        self._wave_h = self.WAVE_H_DEFAULT                              # 波形上下最大宽度（px）
        self._wave_ext_s = self.WAVE_EXT_MS_DEFAULT / 1000.0            # 波形外扩时长（秒）
        self._pit_w = self.PIT_W_DEFAULT                               # PIT 音高线粗细（px）
        # 配色初值（取自 core.project.DEFAULT_COLORS，可由设置→「配色」页覆盖）
        self._note_rgb = _rgb_of(DEFAULT_COLORS["note_color"])
        self._note_fill = QColor(*self._note_rgb, self.NOTE_ALPHA)
        self._note_border = QColor(*self._note_rgb, self.NOTE_BORDER_ALPHA)
        self._pit_rgb = _rgb_of(DEFAULT_COLORS["pit_color"])
        self._pit_color = QColor(*self._pit_rgb, self.PIT_ALPHA)
        self.set_note_colors(DEFAULT_COLORS["note_color"])
        self.set_pit_color(DEFAULT_COLORS["pit_color"])
        self.show_spec = "spec" in layers
        self.show_notes = "notes" in layers
        self.show_wave = True        # 「音符波形」（视图菜单可开关，存工程设置）
        # 音符块**内**的文字（辅音+元音标签）：默认关 —— PIT 曲线穿过音符块时会把块内
        # 文字压住，看不清（既然影响看 PIT，就直接关掉）。同一标签始终写在
        # 词首音符的**块外上方**（见 `_draw_notes`），所以关掉块内文字不丢信息。
        self.show_note_label_inner = False
        self.spec: SpecData | None = None
        self.cmap_name = DEFAULT_CMAP   # soft：默认配色
        self.opacity = 1.0           # 频谱（mel 底图）透明度：固定全不透明（拉杆已移除）
        self.note_opacity = 1.0      # 音符 / F0 透明度：固定全不透明（拉杆已移除）
        self._img: QImage | None = None
        self._img_key = None
        self._f0: np.ndarray | None = None
        self._f0_hop = 0.01
        self._center_midi = 60.0
        self._label_pt = float(self.LABEL_PT)   # 标签字号（set_label_font 可改）
        self._pending_center: float | None = None   # 工程记忆的纵中心（仅首句对音高时生效一次）
        self._drag: dict | None = None
        self._pitch_anchor: tuple[float, float] | None = None   # 琴键列拖拽平移音高
        self.labeler = None            # 音符标签器（主窗口注入：辅音+元音）
        self._pan_midi0 = self._center_midi      # 中键自由平移的起始音高中心
        self.setMinimumHeight(180 if self.show_notes else 96)

    # ---- 中键自由平移：水平改时间窗、垂直改音高中心（波形/层信息条只水平移动） ----
    def _start_pan(self, x: float, y: float = 0.0):
        super()._start_pan(x, y)
        self._pan_midi0 = self._center_midi

    def on_pan_vertical(self, y: float) -> None:
        """钢琴卷帘窗：中键竖直分量 = 上下移动画布（内容跟手）。

        下拖 → 画布内容下移 → 屏幕中央露出更高的音 → 音高中心增大（低音在下）。
        """
        if not self.show_notes or self._pan_anchor is None:   # 频谱窗没有音高纵轴
            return
        y0 = self._pan_anchor[1]
        self._center_midi = max(0.0, min(127.0, self._pan_midi0 + (y - y0) / self.KEY_H))

    # ---- 数据 ----
    def set_model(self, model: EditModel | None):
        super().set_model(model)
        self.state.sel_phone = -1
        self.state.sel_note = -1
        self._drag = None
        self._autofit_pitch()
        self.update()

    def set_wave(self, flag: bool) -> None:
        """「音符波形」开关（视图菜单 → 音符波形，存工程设置）。"""
        self.show_wave = bool(flag)
        self.update()

    def set_note_label_inner(self, flag: bool) -> None:
        """音符块**内**文字开关（**软件级**设置，设置 →「面板」页，默认关）。

        关掉后块内不写标签（PIT 曲线不再被文字挡住），词首块外侧那份标签照旧。
        """
        self.show_note_label_inner = bool(flag)
        self.update()

    def set_spec(self, spec: SpecData | None):
        self.spec = spec
        self._img = None
        self._img_key = None
        self.update()

    def set_f0(self, f0: np.ndarray | None, hop: float = 0.01):
        self._f0 = f0
        self._f0_hop = hop
        self.update()

    def set_cmap(self, name: str):
        if name in CMAPS:
            self.cmap_name = name
        self._img = None
        self._img_key = None
        self.update()

    def _autofit_pitch(self):
        if self.model is None:
            return
        if self._pending_center is not None:    # 工程记忆的纵中心：只作用于首句
            self._center_midi = self._pending_center
            self._pending_center = None
            return
        vals = [n.midi for n in self.model.notes if not n.rest]
        if vals:
            self._center_midi = float(sorted(vals)[len(vals) // 2])

    @property
    def pitch_center(self) -> float:
        """当前音高纵中心（关闭工程时记入工程布局记忆）。"""
        return self._center_midi

    def set_pitch_view(self, row_h=None, center=None) -> None:
        """恢复工程记忆的钢琴窗纵高：行高 KEY_H 与音高纵中心。

        center 只在下一次自动对音高时生效（首句）；之后各句仍按音符中位自动居中，
        避免把某一句话的纵中心硬套到别的句上，把音符顶出画面。
        """
        if row_h:
            self.KEY_H = max(5.0, min(56.0, float(row_h)))
        if center is not None:
            self._pending_center = max(0.0, min(127.0, float(center)))
        self.update()

    def on_row_height_wheel(self, delta, pos):
        """Alt+滚轮 = 音高行高缩放（以鼠标 y 处的音为锚点，使该音保持不动）。

        不排除 x < GUTTER：鼠标停在左侧琴键列上调整行高是最自然的位置，
        排除掉会表现成「这个手势没反应」。
        """
        if not self.show_notes:      # 频谱窗没有音高行高（mel 底图按窗高整体拉伸）
            return
        anchor_midi = self.y_to_midi(pos.y())
        self.KEY_H = max(5.0, min(56.0, self.KEY_H * (1.1 if delta > 0 else 0.9)))
        self._center_midi = anchor_midi - (self._midi_center() - pos.y()) / self.KEY_H

    def _midi_center(self) -> float:
        """钢琴窗绘制区垂直中心（顶部标签带以下）。"""
        return (BAND_H + self.height()) / 2

    def midi_to_y(self, midi: float) -> float:
        return self._midi_center() - (midi - self._center_midi) * self.KEY_H

    def y_to_midi(self, y: float) -> float:
        return self._center_midi + (self._midi_center() - y) / self.KEY_H

    def _rest_y(self) -> float:
        """休止（rest）音符的专用通道 y：贴在面板底边。"""
        return self.height() - self.KEY_H - 2

    def on_plain_wheel(self, delta):
        """滚轮 = 上下滚动音高（内容跟手：上滚 → 露出更高音）。

        一格 ≈ SCROLL_PX_PER_NOTCH 像素 → 换算成半音时自动跟随行高（行高越大，
        每格半音越少），所以不管缩放到哪一档，视觉滚动速度都稳定；同时保证
        放大到极大时一格至少走 1 行，避免「一格只动一点点」。
        夹在有效 MIDI 范围内，避免滚到无穷远后「再也动不了」。
        Ctrl / Shift / Alt 组合由基类 wheelEvent 处理。
        """
        if not self.show_notes:      # 频谱窗没有纵向轴（与波形面板语义一致）
            return
        rows = max(1.0, SCROLL_PX_PER_NOTCH / self.KEY_H)
        self._center_midi = max(0.0, min(127.0, self._center_midi + delta / 120.0 * rows))

    # ---- 音符标签：优先注入的「辅音+元音」标签器，未注入时退回覆盖音素串 ----
    def note_labels(self) -> list[str]:
        if self.model is None:
            return []
        if self.labeler is not None:
            try:
                return list(self.labeler(self.model))
            except Exception:
                pass
        return edit.note_phone_labels(self.model)

    # ---- 频谱底图（方向：低频在底、时间从左到右） ----
    def _build_image(self):
        cw, h = self.plot_w(), self.height()
        plot_h = h - BAND_H                 # 底图只铺标签带以下区域
        # 按「设备像素」建位图：HiDPI（缩放 125%/150%）下若按逻辑像素建图，Qt 会用
        # 双线性平滑放大整张图 → 谐波被抹糊。这里保证 1 设备像素 = 1 位图像素。
        dpr = max(1.0, float(self.devicePixelRatioF()))
        iw = max(2, int(round(cw * dpr)))
        ih = max(2, int(round(plot_h * dpr)))
        key = (round(self.state.t0, 6), round(self.state.span, 6), cw, plot_h,
               self.cmap_name, dpr)
        if self._img is not None and self._img_key == key:
            return
        self._img_key = key
        if self.spec is None or iw < 2 or ih < 2:
            self._img = None
            return
        block, _a, _b = self.spec.slice_px(self.state.t0, self.state.t0 + self.state.span)
        if block.shape[1] < 1:
            self._img = None
            return
        band = _resample_crisp(block, iw, axis=1)     # 时间：缩小=抽点、放大=插值
        band = _resample_crisp(band, ih, axis=0)      # 频率：mel 行位图拉伸
        band = np.clip(band + 0.5, 0, 255).astype(np.uint8)
        # band: 行=频率（行 0 为低频）、列=时间（col 0 = t0）
        # 只翻频率行序 → 低频落在图像底部；时间列必须保持原序（否则左右镜像）
        rgb = np.ascontiguousarray(make_cmap(self.cmap_name)[band][::-1, :])
        img = QImage(rgb.tobytes(), iw, ih, iw * 3, QImage.Format_RGB888).copy()
        img.setDevicePixelRatio(dpr)
        self._img = img

    # ---- 绘制 ----
    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing, False)
        self._paint_bg(p, th.ROLL_BG)
        w, h = self.width(), self.height()
        top, bottom = BAND_H, h
        phones = self.model.phones if self.model else []
        st = self.state

        # 1) 频谱底图（仅频谱窗；透明度可调）
        if self.show_spec:
            self._build_image()
            if self._img is not None:
                p.setOpacity(self.opacity)
                p.setRenderHint(QPainter.SmoothPixmapTransform, False)   # 频谱最近邻，禁用平滑
                p.drawImage(GUTTER, BAND_H, self._img)
                p.setOpacity(1.0)
            else:
                p.setPen(QColor("#667"))
                p.drawText(QRectF(GUTTER, top, self.plot_w(), max(1, bottom - top)),
                           Qt.AlignCenter, "无频谱（等待预处理）")

        p.setRenderHint(QPainter.Antialiasing, True)

        # 2) 音高参考网格（八度线 + 黑键行淡底）—— 仅钢琴卷帘窗有音高纵轴
        if self.show_notes:
            self._draw_pitch_grid(p, top, bottom)

        # 3) 选中音素底色（TG 编辑：所有音素一律平等，不做无声/有声区分）
        #    注：曾按**光标悬停**高亮所在音素，快速浏览时很晃眼，已按要求移除；
        #    只保留「选中」高亮（点击/拖动产生的明确选中）。
        if 0 <= st.sel_phone < len(phones):
            s, e, _m = phones[st.sel_phone]
            x0, x1 = self.t_to_x(s), self.t_to_x(e)
            p.fillRect(QRectF(x0, top, max(1.0, x1 - x0), max(1.0, bottom - top)),
                       QColor(90, 150, 255, 70))

        # 4) 音素边界（与底图/波形严格对齐）
        self._draw_phone_boundaries(p, phones, top, bottom)
        # 4.5) 音素名（面板顶部；深色描边保证频谱亮区可读）
        self._draw_phone_names(p, phones, top)
        si = st.sel_phone
        if 0 <= si < len(phones):
            s, e, _m = phones[si]
            p.setPen(QPen(QColor(255, 210, 90, 210), 1))
            p.drawLine(int(self.t_to_x(s)), top, int(self.t_to_x(s)), bottom)
            p.drawLine(int(self.t_to_x(e)), top, int(self.t_to_x(e)), bottom)

        # 5) 音符（含块内波形预览与音素名）—— 仅钢琴卷帘窗
        if self.show_notes:
            self._draw_notes(p)
            # 6) F0 曲线（画在钢琴卷帘上层）
            self._draw_f0(p)

        # 7) 琴键列（时间刻度已按要求移除）；频谱窗只留左侧留白，保证时间轴对齐
        if self.show_notes:
            self._draw_keys(p, top, bottom)
        else:
            self._draw_gutter(p)
        self._draw_play_cursor(p)
        self._draw_frame(p)
        p.end()

    def _draw_pitch_grid(self, p: QPainter, top: float, bottom: float):
        """音高参考网格（八度线 + 黑键行淡底）—— 只有钢琴卷帘窗有音高纵轴。"""
        w = self.width()
        for midi in self.KEYS:
            y = self.midi_to_y(midi)
            if y < top - self.KEY_H or y > bottom + self.KEY_H:
                continue
            if (midi % 12) in (1, 3, 6, 8, 10):
                p.fillRect(QRectF(GUTTER, y - self.KEY_H / 2, self.plot_w(), self.KEY_H),
                           QColor(0, 0, 0, 26))
            if midi % 12 == 0:
                p.setPen(QPen(QColor(255, 255, 255, 28), 1))
                p.drawLine(GUTTER, int(y), w, int(y))

    def _draw_notes(self, p: QPainter):
        """音符 + 音素标签（休止画在底部休止轨）—— 仅钢琴卷帘窗绘制。

        「波形预览」画在音符**行下方的独立泳道**里（见 `_draw_wave_segments`）：
        先把波形组铺好，再画音符块，故波形永远不会压在音符上。
        """
        w, st = self.width(), self.state
        labels = self.note_labels()
        notes = list(self.model.notes) if self.model else []
        word_starts = set(word_start_notes(notes))   # 词首音符 → 块左上角外侧再写一份标签
        rects: list[tuple[int, object, QRectF]] = []
        for i, n in enumerate(notes):
            x0, x1 = self.t_to_x(n.start), self.t_to_x(n.end)
            if x1 < GUTTER or x0 > w:
                continue
            y = self._rest_y() if n.rest else self.midi_to_y(n.midi)
            h_note = self.KEY_H * 0.78       # 留出缝隙：半音行不相邻，才能看出音分偏移
            rects.append((i, n, QRectF(x0, y - h_note * 0.5, max(1.5, x1 - x0), h_note)))
        # 波形段（一组一组；波形开关关掉时仍算出几何，好让音素名有落脚处）
        bands = self._draw_wave_segments(p, notes)
        self._draw_phone_times(p, bands)
        # 音符块 + 块内合并标签
        p.setOpacity(self.note_opacity)
        for i, n, rect in rects:
            if n.rest:
                fill, border = QColor(120, 120, 132, 190), QColor(160, 160, 172, 200)
            elif n.slur:
                fill, border = QColor(120, 235, 140, 200), QColor(215, 255, 220, 220)
            else:
                # 普通音符：颜色来自工程设置（设置→「配色」页 / 配色方案）
                fill, border = self._note_fill, self._note_border
            p.setBrush(fill)
            p.setPen(QPen(border, 1))
            p.drawRoundedRect(rect, 2, 2)
            if not n.rest:
                # 块正中的水平细线 = 该音符的**绝对 MIDI 音高**（midi_to_y 正好落在块中心），
                # 便于对照 PIT 曲线与音符的音分偏移。颜色与音符边缘同色；休止符不在音高行上，不画。
                ym = rect.center().y()
                p.setPen(QPen(border, 1))
                p.drawLine(QPointF(rect.left() + 1.0, ym), QPointF(rect.right() - 1.0, ym))
            if i == st.sel_note:
                p.setPen(QPen(QColor(255, 235, 90, 250), 2))
                p.setBrush(Qt.NoBrush)
                p.drawRoundedRect(rect.adjusted(-1, -1, 1, 1), 2, 2)
            # 音符内显示「前面的辅音+当前的元音」合并标签（切分多出的音符显示 -）；
            # 逐个音素的名字改写在音符下方的波形带上（「音素时间」）
            txt = labels[i] if i < len(labels) else ""
            if txt and i in word_starts:
                # 词首音符：在块**左上角外侧**再写一份同样的标签 —— 块内文字会被 PIT 曲线
                # 压住看不清，写在块外上方只与上方琴键行的空档相叠，好读。
                f = QFont("Microsoft YaHei", self._label_pt)
                p.setFont(f)
                p.setPen(QColor(235, 244, 252, 235))
                p.drawText(corner_label_pos(rect.top(), QFontMetricsF(f),
                                            max(GUTTER + 2.0, rect.left() + 2.0)), txt)
            if (self.show_note_label_inner and rect.width() > 12 and txt and not n.rest):
                p.setFont(QFont("Microsoft YaHei", self._label_pt))
                p.save()
                p.setClipRect(rect.adjusted(1, 0, -1, 0))
                p.setPen(QColor(10, 20, 30, 230))
                p.drawText(rect, Qt.AlignLeft | Qt.AlignVCenter, " " + txt)
                p.restore()
        p.setOpacity(1.0)
        p.setBrush(Qt.NoBrush)

    def _group_notes(self, notes) -> list[list]:
        """把音符分成「波形段」：按**该段累计音高跨度**（段内最高音 − 最低音）判界。

        判据为什么不用「相邻两音的距离」：一路级进爬升/下降时每一步都很小，
        只看相邻差会让整条连续音阶永远串成一段、波形跟不上旋律线条。
        改看**整段跨度**后，音高走得越远越早断开。

        规则：
          - 段内跨度（最高音 − 最低音）≤ `WAVE_GROUP_MAX_SPAN` 半音 → 继续并入；
            再并入这个音符就会超出跨度 → 另起一段（该音成为新段的最低/最高音）；
          - 休止（rest）＝**连接两边的桥**：音高继承**前一个**有声音符，故不参与跨度
            判定、**永远不会成为分界**，只把前后有声音符连起来画成一条完整波形
            （避免前后两段音高只差音分、各自挂在近乎同一行上画成两条重叠的带子看不清）；
          - 段首的休止没有前驱可继承 → 挂到后面第一个有声段上；
          - 整句全是休止 → 没有音高可挂，不画（见 `_wave_band_rect`）。
        """
        groups: list[list] = []
        pending: list = []                  # 段首的无声音符：留给后面第一个有声音符段
        lo: float | None = None             # 当前段的音高下限（只看有声音符）
        hi: float | None = None             # 当前段的音高上限
        for n in notes:
            if n.rest:
                # 休止继承前一有声音符的音高 = 它天然落在 [lo, hi] 内，故并入即可、
                # 既不改跨度也不分界（这正是「桥」的含义）
                (groups[-1] if groups else pending).append(n)
                continue
            m = float(n.midi)
            if groups and lo is not None:
                nlo, nhi = min(lo, m), max(hi, m)
                if nhi - nlo <= self.WAVE_GROUP_MAX_SPAN:
                    groups[-1].append(n)
                    lo, hi = nlo, nhi
                    continue
            groups.append(pending + [n])    # 跨度装不下 → 另起一段（段首休止并进来）
            pending = []
            lo = hi = m
        if pending and groups:
            groups[-1].extend(pending)      # 结尾的无声音符：挂到最后一段上
        return groups

    def _wave_band_rect(self, g: list) -> tuple[float, float, float, float] | None:
        """一个音符组的波形带几何 (xa, xb, top, h) —— 用整句采样与同一套
        `t_to_x` 浮点时间映射算屏幕列，保证与音素边界杠逐像素对齐。"""
        smp = self._samples
        if smp is None or len(smp) < 2:
            return None
        voiced = [n for n in g if not n.rest]
        if not voiced:                       # 整段全是无声音符：没有音高可挂，不画
            return None
        t0, t1 = g[0].start, g[-1].end
        pw = self.plot_w()
        if pw < 2 or t1 <= t0:
            return None
        # 「错开时多显示一截」：每段首尾各外扩**固定时长** `WAVE_EXT_S`（**不是像素**：
        # 像素外扩会随时间轴缩放变长变短，就不是稳定的「多显示一截」了）。
        # 外扩区画的是相邻时段的真实采样，故相邻两段有重叠区可对照改边界。
        ext = self.WAVE_EXT_S
        xa = max(GUTTER, int(math.floor(self.t_to_x(max(0.0, t0 - ext)))))
        xb = min(self.width(), int(math.ceil(self.t_to_x(t1 + ext))))
        if xb - xa < 3:
            return None
        # 带子顶部 = 该组**最低音**那一行的下沿 + 固定空隙（不贴音符；
        # 挂最低音是为了整段波形都不会被段里任何音符盖住）
        lo_midi = min(n.midi for n in voiced)
        note_bot = self.midi_to_y(lo_midi) + self.KEY_H * 0.78 * 0.5
        top = note_bot + self.WAVE_GAP
        return float(xa), float(xb), float(top), float(self.WAVE_H)

    def _draw_wave_segments(self, p: QPainter, notes) -> list[tuple]:
        """「波形预览」：按「组」画**完整的**波形包络（不是每音符各截一段）。

        与独立波形面板**共用同一套绘制**（`_draw_wave_envelope`），所以形状、振幅比例、
        配色跟上面那个波形面板一模一样：
          - 波形是**整条振幅包络**，安静处收成一根细线、响处涨满整条带子；
          - 带子挂在**整段最低音那一行的下方**、与音符留出空隙（波形并不贴 MIDI 音符）；
          - 带高固定像素（`WAVE_H`，设置面板可改），纵向不随行高/缩放变；
          - 一组一组画（段内**累计音高跨度** ≤ `WAVE_GROUP_MAX_SPAN` 半音的连续音符算一段；
            休止是「桥」、永不分界），
            两段最低音不同就**错开**一层，段首尾做 alpha 渐隐（「淡化」）。
        返回每段的 (xa, xb, top, h, t0, t1) 供音素名定位。
        """
        groups = self._group_notes(notes)
        bands: list[tuple] = []
        for g in groups:
            geo = self._wave_band_rect(g)
            if geo is None:
                continue
            xa, xb, top, h = geo
            t0, t1 = g[0].start, g[-1].end
            bands.append((xa, xb, top, h, t0, t1))
            if not self.show_wave:
                continue
            self._draw_wave_envelope(p, QRectF(xa, top, xb - xa, h), self.WAVE_FADE_PX)
        return bands

    def _draw_phone_times(self, p: QPainter, bands: list[tuple]):
        """「音素时间」：音素名按各自的时段写在波形带上（宽字距、浅白），
        太窄（<14px）不写，去掉语言前缀（zh/a → a）。与波形开关解耦：关掉波形
        也仍留着名字，方便对着波形看发音位置。
        """
        if not bands or self.model is None:
            return
        f = QFont("Microsoft YaHei", self._label_pt)
        f.setLetterSpacing(QFont.AbsoluteSpacing, 2.0)   # 宽字距观感
        p.save()
        p.setFont(f)
        p.setPen(QColor(*self.WAVE_NAME_RGBA))
        for xa, xb, top, h, t0, t1 in bands:
            for s, e, m in self.model.phones:
                if not m:
                    continue
                a, b = max(float(s), t0), min(float(e), t1)
                if b - a <= 1e-6:              # 不在本组时段内
                    continue
                pxa = max(xa, self.t_to_x(a))  # 名字只占自己的时段格，且不越出本段
                pxb = min(xb, self.t_to_x(b))
                if pxb - pxa < 14:             # 太窄放不下就不写，避免糊成一团
                    continue
                p.save()
                p.setClipRect(QRectF(pxa, top, pxb - pxa, h))
                p.drawText(QRectF(pxa, top, pxb - pxa, h),
                           Qt.AlignHCenter | Qt.AlignVCenter, str(m).split("/")[-1])
                p.restore()
        p.restore()

    def _draw_gutter(self, p: QPainter):
        """频谱窗的左侧留白：频谱窗没有音高纵轴，只留与钢琴窗等宽的空白 + 分隔线，
        使两窗的时间轴起点逐像素对齐。"""
        p.fillRect(0, 0, GUTTER, self.height(), QColor("#171a20"))
        p.setPen(QPen(QColor(th.LINE), 1))
        p.drawLine(GUTTER, 0, GUTTER, self.height())

    def _draw_f0(self, p: QPainter):
        f0 = self._f0
        if f0 is None or len(f0) == 0:
            return
        hop = self._f0_hop
        i0 = max(0, int(self.state.t0 / hop))
        i1 = min(len(f0), int((self.state.t0 + self.state.span) / hop) + 1)
        if i1 <= i0:
            return
        pts: list[QPointF | None] = []
        for i in range(i0, i1):
            f = float(f0[i])
            if f <= 0 or not math.isfinite(f):
                pts.append(None)
                continue
            midi = 69.0 + 12.0 * math.log2(f / 440.0)
            pts.append(QPointF(self.t_to_x(i * hop), self.midi_to_y(midi)))
        # PIT 线：细一号少占视觉空间，好让波形 / 音素边界看得更清；
        # 粗细由设置面板的「PIT 音高线粗细」拉杆控制（1.0=最细 / 2.0=最粗），
        # 颜色由设置→「配色」页自定义（默认 #ffcd5a）
        p.setPen(QPen(self._pit_color, self._pit_w))
        run: list[QPointF] = []
        for pt in pts + [None]:
            if pt is None:
                if len(run) > 1:
                    p.drawPolyline(QPolygonF(run))
                run = []
            else:
                run.append(pt)

    def _draw_keys(self, p: QPainter, top: float, bottom: float):
        p.fillRect(0, 0, GUTTER, self.height(), QColor("#171a20"))
        p.setFont(QFont("Consolas", 7))
        for midi in self.KEYS:
            y = self.midi_to_y(midi)
            if y < top - self.KEY_H or y > bottom + self.KEY_H:
                continue
            is_black = (midi % 12) in (1, 3, 6, 8, 10)
            rect = QRectF(0, y - self.KEY_H / 2, GUTTER if not is_black else GUTTER * 0.62,
                          self.KEY_H)
            p.fillRect(rect, QColor("#0f1116" if is_black else "#cfd3da"))
            p.setPen(QColor("#9aa0ab"))
            if midi % 12 == 0:
                p.drawText(QRectF(2, y - self.KEY_H / 2, GUTTER - 4, self.KEY_H),
                           Qt.AlignLeft | Qt.AlignVCenter, f"C{midi // 12 - 1}")
        p.setPen(QPen(QColor(th.LINE), 1))
        p.drawLine(GUTTER, 0, GUTTER, self.height())

    # ---- 交互 ----
    def _note_at(self, t: float, y: float) -> int:
        if self.model is None:
            return -1
        if abs(y - self._rest_y()) <= self.KEY_H:
            ri = self.model.note_index_at(t, 0.0, include_rest=True)
            if ri >= 0 and self.model.notes[ri].rest:
                return ri
        return self.model.note_index_at(t, self.y_to_midi(y), tol_midi=0.6)

    def _note_edge_hit(self, t: float, y: float) -> tuple[int, str] | None:
        if self.model is None:
            return None
        tol = self.sec_per_px() * 5
        midi = self.y_to_midi(y)
        for i, n in enumerate(self.model.notes):
            if abs(n.midi - midi) > 0.6:
                continue
            if i > 0 and abs(t - n.start) <= tol:
                return (i, "start")
            if abs(t - n.end) <= tol:
                return (i, "end")
        return None

    # ---- 右键音符菜单（连音 / 无声）------------------------------------
    def _build_note_menu(self, index: int) -> QMenu:
        """按音符**当前状态**生成菜单：文案给相反动作；休止音符上「连音」置灰。"""
        n = self.model.notes[index]
        m = QMenu(self)
        a_slur = m.addAction("取消连音" if n.slur else "设为连音")
        a_slur.setData("slur")
        a_slur.setEnabled(not n.rest)      # 无声音符不可能连音
        a_rest = m.addAction("取消无声" if n.rest else "设为无声")
        a_rest.setData("rest")
        return m

    def _apply_note_flag(self, index: int, which: str) -> bool:
        """连音 / 无声开关（写入撤销栈）；返回是否真的改了。"""
        if self.model is None or not (0 <= index < len(self.model.notes)):
            return False
        n = self.model.notes[index]
        if which == "slur":
            return edit.set_note_slur(self.model, index, not n.slur)
        if which == "rest":
            return edit.set_note_rest(self.model, index, not n.rest)
        return False

    def _show_note_menu(self, index: int, gpos):
        act = self._build_note_menu(index).exec(gpos)
        if act is not None and self._apply_note_flag(index, act.data()):
            self.sig_changed.emit()
            self.update()

    def mousePressEvent(self, event):
        if self.model is None:
            return
        pos = event.position()
        t = self.x_to_t(pos.x())
        btn = event.button()
        # 右键：①命中音符 → 音符菜单（设为/取消 连音·无声）；②未命中 → 播放所点音素
        # 菜单项文案按**当前状态**给出相反动作，一眼可见该怎么改。
        if btn == Qt.RightButton and pos.x() >= GUTTER:
            ni = self._note_at(t, pos.y()) if self.show_notes else -1
            if ni >= 0:
                self.state.sel_note, self.state.sel_phone = ni, -1
                self.sig_selection.emit()
                self.update()
                self._show_note_menu(ni, event.globalPosition().toPoint())
                return
            idx = self.model.phone_index_at(t)
            if idx >= 0:
                s, e, _m = self.model.phones[idx]
                self.state.sel_phone, self.state.sel_note = idx, -1
                self.sig_selection.emit()
                self.sig_play_phoneme.emit(s, e)
                self.update()
            return
        # 琴键列：左键/中键上下拖拽 = 垂直平移音高（琴键列中键上下移的手势）
        if pos.x() < GUTTER:
            if self.show_notes and btn in (Qt.LeftButton, Qt.MiddleButton):
                self._pitch_anchor = (pos.y(), self._center_midi)
                self.setCursor(Qt.SizeVerCursor)
            return
        if btn == Qt.MiddleButton:                 # 中键平移（水平改时间窗、竖直改音高中心）
            self._start_pan(pos.x(), pos.y())
            self.setCursor(Qt.ClosedHandCursor)
            return
        if btn != Qt.LeftButton:
            return
        self._clear_tier_sel()      # 层信息栏的词/短语高亮一并作废

        # 1) 音符端点（改时间，与相邻音符共享边界 → 总时长不变）；频谱窗无音符层
        hit = self._note_edge_hit(t, pos.y()) if self.show_notes else None
        if hit is not None:
            i, which = hit
            self.state.sel_note, self.state.sel_phone = i, -1
            self._drag = {"kind": "edge", "index": i, "which": which,
                          "before": self.model.snapshot()}
            self.setCursor(Qt.SizeHorCursor)
            self.sig_selection.emit()
            self.update()
            return
        # 2) 音符本体（改音高，自由不限半音）；频谱窗无音符层
        ni = self._note_at(t, pos.y()) if self.show_notes else -1
        if ni >= 0:
            self.state.sel_note, self.state.sel_phone = ni, -1
            self._drag = {"kind": "pitch", "index": ni, "before": self.model.snapshot()}
            self.sig_selection.emit()
            self.update()
            return
        # 3) 音素边界（拖动改对齐，含 SP/AP 等无声音素）
        drag = self._try_boundary_drag(t)
        if drag is not None:
            self._drag = drag
            self.setCursor(Qt.SizeHorCursor)
            return
        # 4) 选中音素
        idx = self.model.phone_index_at(t)
        self.state.sel_phone, self.state.sel_note = idx, -1
        if idx >= 0:
            self.sig_status.emit(f"选中音素 {self.model.phones[idx][2] or '·'}")
        self.sig_selection.emit()
        self.update()

    def mouseMoveEvent(self, event):
        pos = event.position()
        self._set_cursor_t(self.x_to_t(pos.x()) if pos.x() >= GUTTER else -1.0)
        if self._pitch_anchor is not None:      # 琴键列拖拽 → 垂直平移音高
            y0, m0 = self._pitch_anchor
            self._center_midi = m0 + (pos.y() - y0) / self.KEY_H
            self.update()
            return
        if self.model is None:
            self.update()
            return
        if self._update_pan(pos.x(), pos.y()):
            return
        if self._drag is not None:
            d = self._drag
            if d["kind"] == "pitch":
                edit.mut_set_note_pitch(self.model, d["index"], self.y_to_midi(pos.y()))
            elif d["kind"] == "edge":
                edit.mut_move_note_edge(self.model, d["index"], d["which"],
                                        self.x_to_t(pos.x()), snap_tol=0.02,
                                        phone_end_limit=self.state.duration)
            else:
                self._apply_boundary(d, self.x_to_t(pos.x()))
            self.update()
            self.sig_viewport.emit()      # 数据变 → 其余面板同步重绘（消除"挪杠不同步"）
            return
        # 光标位置只留给「播放」工具定位（见 paintEvent 的光标线）；
        # 曾按它反查并高亮所在音素，快速浏览很晃眼，已移除。
        self.update()

    def mouseReleaseEvent(self, event):
        if self._pitch_anchor is not None:
            self._pitch_anchor = None
            self.setCursor(Qt.ArrowCursor)
            return
        if self._end_pan():
            return
        if self._drag is not None:
            label = {"pitch": "修改音高", "edge": "移动音符边界",
                     "boundary": "移动音素边界"}[self._drag["kind"]]
            changed = self.model.commit_snapshot(label, self._drag["before"])
            self._drag = None
            self.setCursor(Qt.ArrowCursor)
            self.update()
            self.sig_viewport.emit()      # 释放后同步其余面板
            if changed:
                self.sig_changed.emit()

    def mouseDoubleClickEvent(self, event):
        """双击 MIDI 区域：命中音符 → 断裂处合并 / 无断裂则切分；
        **双击音符以外的位置（空白/音素区）= 重命名该处音素**。
        频谱窗没有音符层 → 双击一律走重命名音素。"""
        if self.model is None or event.position().x() < GUTTER:
            return
        pos = event.position()
        t = self.x_to_t(pos.x())
        ni = self._note_at(t, pos.y()) if self.show_notes else -1
        if ni >= 0:
            # 1) 断裂合并：命中「相邻且共享边界」的一对音符边界附近即合并。
            #    判定放宽两点（5px 太窄、且只认左侧音符音高）：
            #      a) 共享边界前后各 ±MERGE_TOL_PX 都算命中（站在前/后任一音符上都可触发）
            #      b) 光标音高与这一对里任一音符相符即可（跨界处两声部音高常不同）
            tol = self.sec_per_px() * self.MERGE_TOL_PX
            midi = self.y_to_midi(pos.y())
            notes = self.model.notes
            for i, n in enumerate(notes[:-1]):
                nxt = notes[i + 1]
                if n.rest or nxt.rest:
                    continue
                if abs(nxt.start - n.end) > 1e-3:        # 非共享边界（有断裂才合并）
                    continue
                if abs(t - n.end) > tol:
                    continue
                if abs(n.midi - midi) > 0.6 and abs(nxt.midi - midi) > 0.6:
                    continue
                if self._act_merge(i):
                    self.state.sel_note = i
                    self.sig_selection.emit()
                    return
            # 2) 无断裂 → 切分所在音符
            self._act_split(ni, t)
            self.state.sel_note = ni
            self.sig_selection.emit()
            return
        # 3) 未命中音符 → 重命名所点音素
        idx = self.model.phone_index_at(t)
        if idx >= 0:
            self.state.sel_phone, self.state.sel_note = idx, -1
            self.sig_selection.emit()
            self._rename_phone(idx)

    # ---- 音素动作 ----
    def _rename_phone(self, idx: int):
        old = self.model.phones[idx][2]
        name, ok = QInputDialog.getText(self, "重命名音素", "音素名：", text=old)
        if ok and name != old:
            edit.rename_phone(self.model, idx, name)
            self.sig_changed.emit()
            self.update()

    def _dup_phone(self, idx: int):
        if edit.duplicate_phone(self.model, idx):
            self.sig_changed.emit()
            self.update()

    def _del_phone(self, idx: int):
        if edit.remove_phone(self.model, idx):
            self.state.sel_phone = min(idx, len(self.model.phones) - 1)
            self.sig_changed.emit()
            self.sig_selection.emit()
            self.update()

    # ---- words / texts 条目动作（只在层信息栏里选中了条目时由快捷键触发） ----
    def _tier_sel(self) -> tuple[str, int]:
        """当前层信息栏选中的 words / texts 条目；没选中（或下标已失效）→ ("", -1)。"""
        st = self.state
        if st.sel_tier in ("words", "texts") and st.sel_tier_idx >= 0:
            if st.sel_tier_idx < len(self._tier_items(st.sel_tier)):
                return st.sel_tier, st.sel_tier_idx
            st.sel_tier, st.sel_tier_idx = "", -1
        return "", -1

    def _tier_items(self, tier: str):
        return self.model.words if tier == "words" else self.model.texts

    def _dup_tier(self):
        tier, i = self._tier_sel()
        if edit.duplicate_tier_item(self.model, tier, i):
            self.sig_changed.emit()
            self.update()

    def _del_tier(self):
        tier, i = self._tier_sel()
        if not edit.remove_tier_item(self.model, tier, i):
            return
        if self.state.sel_tier_idx >= len(self._tier_items(tier)):     # 选择前移 / 清空
            self.state.sel_tier_idx = min(i, len(self._tier_items(tier)) - 1)
            if self.state.sel_tier_idx < 0:
                self.state.sel_tier = ""
        self.sig_changed.emit()
        self.sig_selection.emit()
        self.update()

    def _rename_tier(self):
        tier, i = self._tier_sel()
        old = self._tier_items(tier)[i][2]
        title = "重命名词" if tier == "words" else "重命名短语"
        name, ok = QInputDialog.getText(self, title, f"{tier} [{i + 1}] 名称：", text=old)
        if ok and name != old and edit.rename_tier_item(self.model, tier, i, name):
            self.sig_changed.emit()
            self.update()

    # ---- 音符动作 ----
    def _act_split(self, i: int, t: float):
        if edit.split_note(self.model, i, t):
            self.sig_changed.emit()
            self.update()

    def _act_slur(self, i: int):
        if edit.toggle_slur(self.model, i):
            self.sig_changed.emit()
            self.update()

    def _act_merge(self, i: int) -> bool:
        if edit.merge_notes(self.model, i):
            self.sig_changed.emit()
            self.update()
            return True
        return False

    def _act_del_note(self, i: int):
        if edit.delete_note(self.model, i):
            self.state.sel_note = -1
            self.sig_changed.emit()
            self.sig_selection.emit()
            self.update()

    def keyPressEvent(self, event):
        if self.model is None:
            super().keyPressEvent(event)
            return
        key, mods = event.key(), event.modifiers()
        if key == Qt.Key_Space:
            self.sig_play_toggle.emit()
            event.accept()
            return
        if mods & Qt.ControlModifier:
            if key in (Qt.Key_Z, Qt.Key_Y):
                # Ctrl+Y 与 Ctrl+Shift+Z 都是重做，Ctrl+Z 是撤销
                is_redo = key == Qt.Key_Y or bool(mods & Qt.ShiftModifier)
                label = self.model.redo() if is_redo else self.model.undo()
                if label:
                    self.sig_status.emit(("重做：" if is_redo else "撤销：") + label)
                    self.sig_changed.emit()
                    self.update()
                event.accept()
                return
            # 层信息栏里选中了 words / texts 条目 → 快捷键作用于该条目（优先级高于音素）
            if key == Qt.Key_D and self._tier_sel()[0]:
                self._dup_tier()
                event.accept()
                return
            if key == Qt.Key_R and self._tier_sel()[0]:
                self._rename_tier()
                event.accept()
                return
            if key == Qt.Key_D and self.state.sel_phone >= 0:
                self._dup_phone(self.state.sel_phone)
                event.accept()
                return
            if key == Qt.Key_R and self.state.sel_phone >= 0:
                self._rename_phone(self.state.sel_phone)
                event.accept()
                return
        if key == Qt.Key_Delete:
            if self._tier_sel()[0]:
                self._del_tier()
            elif self.state.sel_note >= 0:
                self._act_del_note(self.state.sel_note)
            elif self.state.sel_phone >= 0:
                self._del_phone(self.state.sel_phone)
            event.accept()
            return
        # 方向键：改音高（±1 半音；Shift = ±10 音分微调）
        if key in (Qt.Key_Up, Qt.Key_Down) and self.state.sel_note >= 0:
            step = (10.0 if mods & Qt.ShiftModifier else 100.0) * (1 if key == Qt.Key_Up else -1)
            if edit.nudge_note_pitch(self.model, self.state.sel_note, step):
                self.sig_status.emit(f"音高 {step:+.0f} 音分")
                self.sig_changed.emit()
                self.sig_selection.emit()
                self.update()
            event.accept()
            return
        super().keyPressEvent(event)
