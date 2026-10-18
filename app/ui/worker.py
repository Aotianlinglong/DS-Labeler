# -*- coding: utf-8 -*-
"""
后台推理线程：逐句生成 F0（RMVPE）+ GAME 音符，结果落盘并回传主线程。

- 队列式：可随时 enqueue（支持插队，优先处理当前选中的句子）
- 标准工程：`<根>/.dsl/<句名>.json`（音素 + 音符）、`<根>/.dsl/f0/<句名>.npz`（F0）
- 翻新工程：**不跑 GAME**（音符直接取自 csv），只算 F0 → 缓存落软件根
  `refurbish/cache/<工程名>/f0/<句名>.npz`
- 已存在缓存则秒过，不重跑
"""
from __future__ import annotations

import collections
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import soundfile as sf
from PySide6.QtCore import QThread, Signal

from core.curve import FPS as CURVE_FPS, MouthCurveExtractor, save_curve
from core.paths import GAME_MODEL_DIR, MOUTH_MODEL, RMVPE_MODEL
from core.sentence import SentenceData
from core.textgrid_io import parse_textgrid
from engine.f0 import F0Extractor, save_f0
from engine.game import GameNoteAligner
from engine.onnx_util import last_provider


class InferenceWorker(QThread):
    sig_progress = Signal(str, int, int)      # 当前句名, 已完成, 总数
    sig_sentence = Signal(str, object)        # 句名, SentenceData
    sig_failed = Signal(str, str)             # 句名, 错误
    sig_provider = Signal(str)                # 实际使用的推理设备
    sig_idle = Signal()                       # 队列处理完毕

    def __init__(self, project, parent=None):
        super().__init__(parent)
        self.project = project
        self._queue: collections.deque[str] = collections.deque()
        self._extractor: F0Extractor | None = None
        self._aligner: GameNoteAligner | None = None
        self._classifier = None
        self._pool: ThreadPoolExecutor | None = None    # F0 侧线程（与 GAME 并行）
        self._announced = False
        self._warmed = False
        self._stop = False

    # ---- 队列管理（主线程调用） ---------------------------------======
    def enqueue(self, names, front: bool = False, skip_cached: bool = True) -> int:
        if skip_cached:
            names = [n for n in names if not self._is_cached(n)]
        names = [n for n in names if n]
        if not names:
            return 0
        if front:
            for n in reversed(names):
                if n in self._queue:
                    self._queue.remove(n)
                self._queue.appendleft(n)
        else:
            for n in names:
                if n not in self._queue:
                    self._queue.append(n)
        self._stop = False
        if not self.isRunning():
            self.start()
        return len(names)

    def stop(self) -> None:
        self._stop = True
        self._queue.clear()

    def set_project(self, project) -> None:
        """切换工程：清空队列、重置分类器（模型会话复用，无需重建）。"""
        self._queue.clear()
        self.project = project
        self._classifier = None

    def warmup(self) -> None:
        """预建会话 + 跑一次哑推理（打开工程后调用）。

        ONNX 首次运行要「建会话 + DML 图编译」，一次性约 3s（RMVPE 1s、
        GAME 2s）。把它挪到打开工程后的空闲时间，第一句就不会额外卡住。
        """
        if self.isRunning() or self._warmed:
            return                      # 一个 worker 只预热一次（切排序/刷新列表会重复调）
        self._warmed = True
        self._stop = False
        self.start()

    def busy(self) -> bool:
        return self.isRunning()

    def _is_cached(self, name: str) -> bool:
        p = self.project
        f0 = (p.f0_dir / f"{name}.npz").exists()
        if p.is_refurbish:
            # 翻新：音符直接取自 csv（不做 GAME）→ 只有 F0 需要缓存
            return f0
        return p.sentence_work_file(name).exists() and f0

    # ---- 线程主体 -----------------------------------------------------
    def run(self) -> None:
        done = 0
        self._warmed = True
        self._pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="dsl-f0")
        try:
            self._warmup()
            while self._queue and not self._stop:
                name = self._queue.popleft()
                try:
                    sd = self._process(name)
                    self.sig_sentence.emit(name, sd)
                except Exception as e:  # 单句失败不拖垮整批
                    self.sig_failed.emit(name, f"{type(e).__name__}: {e}")
                done += 1
                self.sig_progress.emit(name, done, done + len(self._queue))
        finally:
            self._pool.shutdown(wait=True)
            self._pool = None
        if done:                       # 空队列（warmup 用）不发「预处理完成」
            self.sig_idle.emit()

    def _warmup(self) -> None:
        """预建会话 + 哑推理。`_stop`（用户关窗）时立刻放弃，别拖着进程退出。"""
        if self._stop:
            return
        try:
            self._ensure_models()
            if self._stop:
                return
            silence = np.zeros(16000, dtype=np.float32)
            self._extractor.extract(silence, 16000)
            if self.project.is_refurbish:
                return                     # 翻新工程不跑 GAME → 无需预热对齐器
            self._aligner.align(
                np.zeros(88200, dtype=np.float32), 44100,
                [(0.0, 0.5, "SP"), (0.5, 1.0, "zh/a")], [2],
                uv_vocab=None, lang=self.project.main_lang)
        except Exception:
            pass                       # 哑推理失败无所谓，真推理时再报

    # ---- 单句处理 -----------------------------------------------------
    def _process(self, name: str) -> SentenceData:
        proj = self.project
        wav_path = proj.wav_file(name)
        work_path = proj.sentence_work_file(name)
        f0_path = proj.f0_dir / f"{name}.npz"

        sd: SentenceData | None = None
        if work_path.exists():
            try:
                sd = SentenceData.load(work_path, name)
            except Exception:
                sd = None

        if proj.is_refurbish:
            # 翻新：音符直接取自 csv（不做 GAME 推理），只需补 F0
            if sd is None:
                sd = proj.source_sentence(name) or SentenceData(name=name, phones=[], notes=[])
            if not f0_path.exists():
                audio, sr = self._read_wav(wav_path)
                self._ensure_models()
                self._f0_to_cache(audio, sr, f0_path)
            return sd

        need_notes = sd is None or not sd.notes
        need_f0 = not f0_path.exists()
        if not need_notes and not need_f0:
            return sd

        audio, sr = self._read_wav(wav_path)
        if sd is None:
            sd = SentenceData(name=name, phones=self._phones_from_tg(name), notes=[])
        if need_notes or need_f0:
            self._ensure_models()      # 会话提前建好（避免两线程同时懒加载）

        # F0 与 GAME **并行**：F0 走 CPU、GAME 走 GPU，互不抢资源，
        # 单句墙钟从 (1.0 + 1.1)s 降到 ≈ max(1.0, 1.1)s。
        fut = None
        if need_f0:
            if self._pool is not None:
                fut = self._pool.submit(self._f0_to_cache, audio, sr, f0_path)
            else:                      # 非 run() 上下文（测试）→ 同步兜底
                self._f0_to_cache(audio, sr, f0_path)
        if need_notes:
            sd.notes = self._game_notes(audio, sr, sd.phones)
            sd.save(work_path)
        if fut is not None:
            fut.result()               # 等 F0 落盘；异常在此抛出（单句失败不拖垮整批）
        return sd

    def _f0_to_cache(self, audio: np.ndarray, sr: int, f0_path) -> None:
        f0, uv = self._extractor.extract(audio, sr)
        save_f0(f0_path, f0, uv)

    def _phones_from_tg(self, name: str) -> list[tuple[float, float, str]]:
        tg = self.project.tg_dir / f"{name}.TextGrid"
        sd = parse_textgrid(tg)
        return [(float(a), float(b), str(m))
                for a, b, m in zip(sd.phones.xmins, sd.phones.xmaxs, sd.phones.marks)]

    def _read_wav(self, path) -> tuple[np.ndarray, int]:
        x, sr = sf.read(str(path), dtype="float32", always_2d=True)
        if x.shape[1] > 1:
            x = x.mean(axis=1)
        else:
            x = x[:, 0]
        return np.ascontiguousarray(x, dtype=np.float32), int(sr)

    def _ensure_models(self) -> None:
        if self._extractor is None:
            # RMVPE 走 **CPU**：9s 音频 CPU 0.23s / DML 0.88s
            # （rmvpe.onnx 362MB，DML 把算子切来切去 + GPU↔CPU 拷贝，反而慢 4 倍）。
            # GAME 仍是 GPU 快（8 步 segmenter：DML 127ms/步 vs CPU 247ms/步）。
            self._extractor = F0Extractor(RMVPE_MODEL, prefer_gpu=False)
        if self.project.is_refurbish:
            return                     # 翻新工程不做 GAME → 不加载对齐器 / 分类器
        if self._aligner is None:
            self._aligner = GameNoteAligner(GAME_MODEL_DIR, prefer_gpu=True)
        if self._classifier is None:
            self._classifier = self.project.make_classifier()

    def _game_notes(self, audio: np.ndarray, sr: int,
                    phones: list[tuple[float, float, str]]):
        self._ensure_models()
        if not self._announced:
            self.sig_provider.emit(last_provider())
            self._announced = True
        ph_seq = [m for _, _, m in phones]
        if not ph_seq:
            return []
        ph_num = self._classifier.compute_ph_num(ph_seq)
        return self._aligner.align(
            audio, sr, phones, ph_num,
            uv_vocab=set(self.project.uv_vocab),
            lang=self.project.main_lang,
        )


class MouthCurveWorker(QThread):
    """后台计算**口型曲线**（R3MOE）：逐句推理 → 落缓存 `<curve_dir>/<句名>.npz` → 回传主线程。

    与 InferenceWorker 分开的原因：口型是**独立可关的面板**，默认不开；开了才建会话/算缓存，
    彼此不互相牵制（「批量预处理」不该顺带算出没人看的口型）。两个入口共用同一个线程：
      · 切句时缺缓存且面板可见 → enqueue([当前句], front=True)
      · 菜单「计算所有文件口型」    → enqueue(全部句名)
    已有缓存默认跳过；权重缺失时直接报一次失败并清空队列（不逐句刷屏）。

    走 **CPU**：小模型（约 40MB）+ 双层 BiLSTM，CPU 足够；也避免和 GAME 抢 DML 显存。
    """

    sig_progress = Signal(str, int, int)      # 当前句名, 已完成, 总数
    sig_curve = Signal(str, object, float)    # 句名, np.ndarray, fps
    sig_failed = Signal(str, str)             # 句名, 错误
    sig_idle = Signal()                       # 队列处理完毕

    def __init__(self, project, parent=None):
        super().__init__(parent)
        self.project = project
        self._queue: collections.deque[str] = collections.deque()
        self._stop = False

    # ---- 队列管理（主线程调用） ---------------------------------------
    def enqueue(self, names, front: bool = False, skip_cached: bool = True) -> int:
        names = [n for n in names if n]
        if skip_cached:
            names = [n for n in names if not self._is_cached(n)]
        if not names:
            return 0
        if front:
            for n in reversed(names):
                if n in self._queue:
                    self._queue.remove(n)
                self._queue.appendleft(n)
        else:
            for n in names:
                if n not in self._queue:
                    self._queue.append(n)
        self._stop = False
        if not self.isRunning():
            self.start()
        return len(names)

    def stop(self) -> None:
        self._stop = True
        self._queue.clear()

    def set_project(self, project) -> None:
        self._queue.clear()
        self.project = project

    def busy(self) -> bool:
        return self.isRunning()

    def _curve_path(self, name: str):
        return self.project.curve_dir / f"{name}.npz"

    def _is_cached(self, name: str) -> bool:
        return self._curve_path(name).exists()

    # ---- 线程主体 -----------------------------------------------------
    def run(self) -> None:
        done = 0
        extractor = MouthCurveExtractor(MOUTH_MODEL)
        if not extractor.available():
            # 权重缺失（用户未自备）：报一次就好，别逐句刷错
            self.sig_failed.emit(self._queue[0] if self._queue else "",
                                 f"缺少口型模型权重: {MOUTH_MODEL}")
            self._queue.clear()
            self.sig_idle.emit()
            return
        while self._queue and not self._stop:
            name = self._queue.popleft()
            try:
                curve = self._process(name, extractor)
                self.sig_curve.emit(name, curve, CURVE_FPS)
            except Exception as e:              # 单句失败不拖垮整批
                self.sig_failed.emit(name, f"{type(e).__name__}: {e}")
            done += 1
            self.sig_progress.emit(name, done, done + len(self._queue))
        if done:
            self.sig_idle.emit()

    def _process(self, name: str, extractor: MouthCurveExtractor) -> np.ndarray:
        wav_path = self.project.wav_file(name)
        x, sr = sf.read(str(wav_path), dtype="float32", always_2d=True)
        x = x.mean(axis=1) if x.shape[1] > 1 else x[:, 0]
        curve = extractor.extract(np.ascontiguousarray(x, dtype=np.float32), int(sr))
        save_curve(self._curve_path(name), curve, CURVE_FPS)
        return curve
