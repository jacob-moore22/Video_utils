"""Pure data-oriented core: file loading, scene state, rendering, and video export.

No Qt or other GUI dependencies live in this package, so the same functions can
be driven by any frontend (PySide6 today, possibly Trame later) and packaged
cleanly by PyInstaller.
"""
