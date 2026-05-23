import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

CONFIGS = [
    (30, 0), (30, 1), (30, 2),
    (60, 0), (60, 1), (60, 2),
]


def main() -> None:
    for n, seed in CONFIGS:
        cmd = [
            sys.executable,
            "scripts/make_splits.py",
            "--images-per-class", str(n),
            "--seed", str(seed),
        ]
        print(f"$ {' '.join(cmd)}")
        subprocess.run(cmd, check=True, cwd=ROOT)
    print("\nAll splits generated.")


if __name__ == "__main__":
    main()
