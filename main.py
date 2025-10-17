#!/usr/bin/env python3
import os
import sys
import shutil

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
VENV_DIR = os.path.join(PROJECT_ROOT, ".venv")
VENV_PY = os.path.join(VENV_DIR, "Scripts", "python.exe") if os.name == "nt" else os.path.join(VENV_DIR, "bin", "python")


def _in_this_venv() -> bool:
    try:
        # When inside a venv, sys.prefix points to the venv root
        return os.path.realpath(sys.prefix).startswith(os.path.realpath(VENV_DIR))
    except Exception:
        return False


def main():
    # If .venv exists and we're not using it, re-exec with the venv's python
    if os.path.isdir(VENV_DIR) and shutil.which(VENV_PY) and not _in_this_venv():
        print(f"[boot] Re-executing under project .venv: {VENV_PY}")
        os.execv(VENV_PY, [VENV_PY, os.path.join(PROJECT_ROOT, "mvp_agent.py"), *sys.argv[1:]])

    # Already in the right interpreter (or no .venv found) — run the agent directly
    os.execv(sys.executable, [sys.executable, os.path.join(PROJECT_ROOT, "mvp_agent.py"), *sys.argv[1:]])


if __name__ == "__main__":
    main()
