"""Entry point for the Solver Viewer GUI.

Usage:
    python vtk_to_video.py [path/to/file.pvd|.vtm|.vtu|...]
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


def _configure_qt_platform() -> None:
    """Force xcb under Wayland so VTK's X11 OpenGL render window can attach.

    VTK 9.x still ships only the X11 backend on Linux; running under Qt's
    native Wayland backend reaches it through XWayland in a half-broken way
    and crashes with `vtkXOpenGLRenderWindow ... The result is out of range`
    followed by `BadWindow (invalid Window parameter)`. Forcing xcb makes Qt
    use X11 directly (still via XWayland on a Wayland session, but in a path
    VTK understands), which fixes the reparenting failure.

    Respect any user-set QT_QPA_PLATFORM so power users can override.
    """
    if sys.platform.startswith("linux") and "QT_QPA_PLATFORM" not in os.environ:
        if os.environ.get("WAYLAND_DISPLAY") or os.environ.get("XDG_SESSION_TYPE") == "wayland":
            os.environ["QT_QPA_PLATFORM"] = "xcb"


_configure_qt_platform()


from PySide6.QtCore import Qt, QCoreApplication  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from gui.main_window import MainWindow  # noqa: E402


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="PyVista-based solver result viewer.")
    parser.add_argument(
        "file",
        nargs="?",
        type=Path,
        help="Optional VTK file to open on startup (.pvd, .vtm, .vtu, .vtk, .vts, .vtr, .vtp).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv if argv is not None else sys.argv[1:])

    QCoreApplication.setAttribute(Qt.AA_ShareOpenGLContexts, True)

    app = QApplication(sys.argv)
    window = MainWindow(initial_file=args.file)
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
