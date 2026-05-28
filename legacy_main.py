from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import sys


PROJECT_ROOT = Path(__file__).resolve().parent
LEGACY_ROOT = PROJECT_ROOT / "legacy_workflows" / "legacy_synthetic_optimization"

ENTRYPOINTS = {
    "synthetic": "main.py",
    "nsga2": "run_nsga2.py",
    "real-quick": "run_real_quick.py",
    "real-stress": "run_real_stress.py",
}


def _default_python() -> Path:
    if os.name == "nt":
        venv_python = PROJECT_ROOT / ".venv" / "Scripts" / "python.exe"
    else:
        venv_python = PROJECT_ROOT / ".venv" / "bin" / "python"
    return venv_python if venv_python.exists() else Path(sys.executable)


def _legacy_env() -> dict[str, str]:
    env = os.environ.copy()
    legacy_path = str(LEGACY_ROOT)
    current_pythonpath = env.get("PYTHONPATH")
    env["PYTHONPATH"] = legacy_path if not current_pythonpath else legacy_path + os.pathsep + current_pythonpath
    return env


def run_legacy(mode: str, legacy_args: list[str]) -> int:
    if not LEGACY_ROOT.exists():
        raise FileNotFoundError(f"Legacy project directory not found: {LEGACY_ROOT}")

    script_name = ENTRYPOINTS[mode]
    script_path = LEGACY_ROOT / script_name
    if not script_path.exists():
        raise FileNotFoundError(f"Legacy entrypoint not found: {script_path}")

    command = [str(_default_python()), str(script_path), *legacy_args]
    completed = subprocess.run(command, cwd=LEGACY_ROOT, env=_legacy_env(), check=False)
    return int(completed.returncode)


def parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(
        description="Run archived legacy synthetic optimization workflows without affecting main.py."
    )
    parser.add_argument(
        "--mode",
        choices=sorted(ENTRYPOINTS),
        default="synthetic",
        help="Legacy workflow to run. Default: synthetic.",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List available legacy workflows and exit.",
    )
    return parser.parse_known_args()


def main() -> None:
    args, legacy_args = parse_args()
    if legacy_args[:1] == ["--"]:
        legacy_args = legacy_args[1:]

    if args.list:
        print("Available legacy workflows:")
        for mode, script_name in sorted(ENTRYPOINTS.items()):
            print(f"- {mode}: {LEGACY_ROOT / script_name}")
        return

    raise SystemExit(run_legacy(args.mode, legacy_args))


if __name__ == "__main__":
    main()
