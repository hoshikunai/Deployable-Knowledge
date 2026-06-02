import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def get_python_command() -> str:
    if os.environ.get("DK_PYTHON"):
        return os.environ["DK_PYTHON"]

    venv_python = (
        ROOT / "venv" / "Scripts" / "python.exe"
        if sys.platform == "win32"
        else ROOT / "venv" / "bin" / "python"
    )

    if venv_python.exists():
        return str(venv_python)

    return sys.executable


def main() -> int:
    python = get_python_command()
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT)

    try:
        result = subprocess.run(
            [
                python,
                "-m",
                "PyInstaller",
                "--clean",
                "--noconfirm",
                "electron/backend.spec",
            ],
            cwd=ROOT,
            env=env,
        )

    except OSError as error:
        print(error, file=sys.stderr)
        return 1

    if result.returncode != 0:
        print(
            "Backend build failed. Install build dependencies with: "
            "python -m pip install -r requirements-build.txt",
            file=sys.stderr,
        )
        return result.returncode or 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
