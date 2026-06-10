from __future__ import annotations

from pathlib import Path
import subprocess
import sys


def _use_project_venv_if_available() -> None:
    # 承载当前函数的核心处理逻辑。
    """检测并切换到项目虚拟环境中的 Python 解释器。"""
    venv_python = Path(__file__).resolve().parent / ".venv" / "Scripts" / "python.exe"
    # 根据条件选择对应处理路径。
    if not venv_python.exists():
        return
    # 根据条件选择对应处理路径。
    if Path(sys.executable).resolve() == venv_python.resolve():
        return
    completed = subprocess.run([str(venv_python), *sys.argv], check=False)
    sys.exit(completed.returncode)


_use_project_venv_if_available()

from experiments.exp_main import main


if __name__ == "__main__":
    main()
