"""Post-OTA relauncher: wait for the old controller to exit, then start a fresh one.

Spawned detached by the controller after a code update is applied. Waiting for the
old PID to die avoids the singleton-broadcaster clash that hangs a 2nd instance.

Usage: python scripts/relaunch_controller.py <old_pid> <project_root>
"""
import os
import subprocess
import sys
import time


def _alive(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        if os.name == "nt":
            import ctypes

            PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
            handle = ctypes.windll.kernel32.OpenProcess(
                PROCESS_QUERY_LIMITED_INFORMATION, False, pid
            )
            if not handle:
                return False
            exit_code = ctypes.c_ulong()
            ctypes.windll.kernel32.GetExitCodeProcess(handle, ctypes.byref(exit_code))
            ctypes.windll.kernel32.CloseHandle(handle)
            return exit_code.value == 259  # STILL_ACTIVE
        os.kill(pid, 0)
        return True
    except Exception:
        return False


def main() -> None:
    old_pid = int(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1].isdigit() else 0
    root = sys.argv[2] if len(sys.argv) > 2 else os.getcwd()

    # Wait for the old controller to fully exit (up to ~30s), then a short grace.
    for _ in range(60):
        if not _alive(old_pid):
            break
        time.sleep(0.5)
    time.sleep(1.5)

    main_py = os.path.join(root, "controller", "main.py")
    subprocess.Popen([sys.executable, main_py], cwd=root)


if __name__ == "__main__":
    main()
