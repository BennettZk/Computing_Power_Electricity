from __future__ import annotations

import os
from pathlib import Path
import sys


def _use_project_venv_if_available() -> None:
    venv_python = Path(__file__).resolve().parent / ".venv" / "Scripts" / "python.exe"
    if not venv_python.exists():
        return
    if Path(sys.executable).resolve() == venv_python.resolve():
        return
    os.execv(str(venv_python), [str(venv_python), *sys.argv])


_use_project_venv_if_available()

from experiments.exp_main import main


if __name__ == "__main__":
    main()
