# -*- coding: utf-8 -*-
"""
DS-Labeler 入口
DiffSinger 数据集标注工具 —— TG 音素标记 + 音高（MIDI/音符）标记一体化软件

环境策略：复用内嵌 Python（旧版 Qt 绑定 + onnxruntime-directml + librosa 等），
GAME 推理使用 ONNX 模型 + onnxruntime-directml（GPU），无需 PyTorch。
"""
import os
import sys

# ---- 环境引导 -------------------------------------------------------------
# 内嵌 Python 是嵌入式分发：存在 python312._pth 时 PYTHONPATH 被忽略，
# 因此 DS-Labeler 自带的 site-packages（后续补装的依赖）必须在入口手动注入。
_APP_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT_DIR = os.path.dirname(_APP_DIR)
# 嵌入式分发不会把脚本所在目录加入 sys.path，必须手动注入 app/ 自身
if _APP_DIR not in sys.path:
    sys.path.insert(0, _APP_DIR)
_SP_DIR = os.path.join(_ROOT_DIR, "site-packages")
if os.path.isdir(_SP_DIR) and _SP_DIR not in sys.path:
    sys.path.insert(0, _SP_DIR)

# 工程内资源路径（引擎、模型等），供各模块引用
ASSETS_DIR = os.path.join(_ROOT_DIR, "assets")
ENGINE_DIR = os.path.join(_ROOT_DIR, "engine")
MODELS_DIR = os.path.join(_ROOT_DIR, "models")

# ---- 关键导入顺序（务必保持）---------------------------------------------
# onnxruntime 必须先于旧版 Qt 绑定导入：旧版 Qt 绑定会加载旧版 VC 运行时 DLL，与
# onnxruntime_pybind11_state 冲突；若在其后导入会报「DLL 初始化例程失败」。
# RMVPE/GAME 推理都依赖 onnxruntime-directml，须先于 Qt 绑定导入。
try:
    import onnxruntime  # noqa: F401
except Exception as _e:  # 缺 onnxruntime 时不阻塞 UI 启动，推理入口再提示
    print(f"[warn] onnxruntime 不可用，推理功能将不可用：{_e}")


def _install_excepthook() -> None:
    """未捕获异常（含 Qt 槽函数内）写进 .dsl/run.log，避免静默崩溃丢数据。"""
    import traceback

    def _hook(etype, value, tb):
        text = "".join(traceback.format_exception(etype, value, tb))
        traceback.print_exc()
        try:
            import datetime
            log = os.path.join(_ROOT_DIR, ".dsl", "run.log")
            os.makedirs(os.path.dirname(log), exist_ok=True)
            ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            with open(log, "a", encoding="utf8") as f:
                f.write(f"[{ts}] [crash]\n{text}\n")
        except Exception:
            pass
        sys.__excepthook__(etype, value, tb)

    sys.excepthook = _hook


def main() -> int:
    from PySide6.QtWidgets import QApplication
    from ui.main_window import MainWindow
    from ui.theme import apply_theme

    app = QApplication(sys.argv)
    app.setApplicationName("DS-Labeler")
    app.setOrganizationName("DS-Labeler")
    apply_theme(app)

    win = MainWindow()
    win.show()

    return app.exec()


if __name__ == "__main__":
    _install_excepthook()
    sys.exit(main())
