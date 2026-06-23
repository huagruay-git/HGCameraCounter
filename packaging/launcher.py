"""
Single-executable entry point for the frozen (.exe) build.

  HGCameraCounter.exe              -> controller GUI (default)
  HGCameraCounter.exe --runtime    -> counting runtime (processor)
  HGCameraCounter.exe --recorder   -> clip recorder

The controller starts the runtime/recorder by re-invoking THIS same exe with the
flag (see controller/main.py _processor_command/_recorder_command), so the heavy
deps (torch/ultralytics/PySide6) are bundled once. We use runpy so the existing
`if __name__ == "__main__"` blocks run unchanged.
"""
import os
import runpy
import sys
from pathlib import Path

# Make stdout/stderr tolerant of non-ASCII so a stray non-UTF-8 print can't crash the
# frozen app on a non-UTF-8 Windows console (e.g. Thai cp874).
for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

# Frozen: resolve relative data/, models/, config next to the exe.
if getattr(sys, "frozen", False):
    _app_dir = Path(sys.executable).resolve().parent
    try:
        os.chdir(str(_app_dir))
    except Exception:
        pass
    if str(_app_dir) not in sys.path:
        sys.path.insert(0, str(_app_dir))


def main() -> None:
    argv = sys.argv[1:]
    # Strip our dispatch flags so the target module's own arg parsing isn't confused.
    _modes = {"--runtime", "--recorder", "--model-ota"}
    sys.argv = [sys.argv[0]] + [a for a in argv if a not in _modes]
    if "--runtime" in argv:
        runpy.run_module("runtime.processor", run_name="__main__")
    elif "--recorder" in argv:
        runpy.run_module("runtime.recorder", run_name="__main__")
    elif "--model-ota" in argv:
        # Pull model/config OTA from Supabase (run on a schedule on the device).
        runpy.run_module("runtime.model_ota", run_name="__main__")
    else:
        runpy.run_module("controller.main", run_name="__main__")


if __name__ == "__main__":
    main()
