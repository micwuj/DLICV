from pathlib import Path

from torchvision.datasets import OxfordIIITPet

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw"


def main() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Downloading Oxford-IIIT Pets to {RAW_DIR} ...")
    OxfordIIITPet(root=str(RAW_DIR), split="trainval", download=True)
    OxfordIIITPet(root=str(RAW_DIR), split="test", download=True)
    print("Done.")


if __name__ == "__main__":
    main()
