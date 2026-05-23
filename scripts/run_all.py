"""Batch runner — uruchamia wiele configow eksperymentow sekwencyjnie.

Argumenty:
    --pattern   glob dla configs/exp/ (np. 'A_*.yaml', 'A_resnet18_*.yaml')
    --overrides dodatkowe configi do merge (np. configs/colab.yaml)
    --dry-run   tylko pokazuje co bedzie uruchomione
    --skip-existing  pomija runy ktorych final_results.json juz istnieje

Przyklady:
    # Wszystkie 24 configi z fazy 2 na Colab T4
    python scripts/run_all.py --overrides configs/colab.yaml

    # Tylko wariant A
    python scripts/run_all.py --pattern 'A_*.yaml' --overrides configs/colab.yaml

    # Wszystkie modele dla seeda 0
    python scripts/run_all.py --pattern '*_seed0.yaml' --overrides configs/colab.yaml

    # Pomin runy juz wykonane (np. po przerwaniu sesji Colab)
    python scripts/run_all.py --overrides configs/colab.yaml --skip-existing
"""
import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
EXP_DIR = ROOT / "configs" / "exp"
BASE = ROOT / "configs" / "base.yaml"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pattern", default="*.yaml")
    ap.add_argument("--overrides", nargs="*", default=[],
                    help="Configi merged miedzy base a exp (np. configs/colab.yaml)")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--skip-existing", action="store_true")
    ap.add_argument("--output-dir", default="outputs")
    args = ap.parse_args()

    configs = sorted(EXP_DIR.glob(args.pattern))
    if not configs:
        print(f"Brak configow dla wzorca: {args.pattern}")
        return

    # Filtr "skip-existing"
    runnable = []
    skipped = []
    for c in configs:
        run_name = c.stem
        final_json = ROOT / args.output_dir / run_name / "final_results.json"
        if args.skip_existing and final_json.exists():
            skipped.append(c)
        else:
            runnable.append(c)

    print(f"Znaleziono {len(configs)} configow.")
    if skipped:
        print(f"Pomijam {len(skipped)} (final_results.json juz istnieje):")
        for c in skipped:
            print(f"  {c.name}")
    print(f"Do uruchomienia: {len(runnable)}")
    for c in runnable:
        print(f"  {c.name}")

    if args.dry_run or not runnable:
        return

    results: list[tuple[str, int, float]] = []
    total_start = time.time()
    for i, cfg in enumerate(runnable, 1):
        elapsed_total = time.time() - total_start
        print(f"\n{'=' * 72}")
        print(f"[{i}/{len(runnable)}] {cfg.name}  (total elapsed: {elapsed_total / 60:.1f} min)")
        print(f"{'=' * 72}")
        start = time.time()
        cmd = [sys.executable, "-m", "src.train", "--config", str(BASE)]
        for ov in args.overrides:
            cmd.append(str(Path(ov)))
        cmd.append(str(cfg))
        rc = subprocess.run(cmd, cwd=ROOT).returncode
        elapsed = time.time() - start
        results.append((cfg.name, rc, elapsed))
        print(f"\n[{cfg.name}] rc={rc} elapsed={elapsed:.0f}s")

    # Summary
    print(f"\n{'=' * 72}\nSUMMARY ({(time.time() - total_start) / 60:.1f} min total)")
    print(f"{'=' * 72}")
    n_fail = 0
    for name, rc, elapsed in results:
        status = "OK  " if rc == 0 else f"FAIL"
        print(f"  {status}  {elapsed:5.0f}s  {name}")
        if rc != 0:
            n_fail += 1
    print(f"\n{len(results) - n_fail}/{len(results)} OK, {n_fail} failed")
    sys.exit(1 if n_fail else 0)


if __name__ == "__main__":
    main()
