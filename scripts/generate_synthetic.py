import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.synth.generate import GenerateConfig, build_pipeline, generate_for_breed
from src.synth.prompts import BREEDS, iter_prompt_variations
from src.utils.device import device_info, get_device
from src.utils.logger import CSVLogger


def count_existing(breed_dir: Path, breed: str) -> int:
    if not breed_dir.exists():
        return 0
    return len(list(breed_dir.glob(f"{breed}_*.png")))


def build_prompt_seed_pairs(breed: str, mode: str, n: int, start_idx: int) -> tuple[list[str], list[int]]:
    pool = iter_prompt_variations(breed, mode=mode)
    prompts = [pool[(start_idx + i) % len(pool)] for i in range(n)]
    seeds = [start_idx + i for i in range(n)]
    return prompts, seeds


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", choices=["D", "E"], required=True)
    ap.add_argument("--n-per-class", type=int, default=300)
    ap.add_argument("--breeds", nargs="*", default=None,
                    help="Subset breeds (default: all 10)")
    ap.add_argument("--output-root", default="data/synthetic",
                    help="Root dir; final path is <root>/variant_<X>/<breed>/")
    ap.add_argument("--lora-dir", default=None,
                    help="Required for --variant E. Expects <breed>.safetensors files")
    ap.add_argument("--image-size", type=int, default=512)
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--guidance", type=float, default=7.5)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--model-id", default="runwayml/stable-diffusion-v1-5")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    breeds = args.breeds or list(BREEDS.keys())
    unknown = [b for b in breeds if b not in BREEDS]
    if unknown:
        print(f"unknown breeds: {unknown}; valid: {list(BREEDS.keys())}")
        sys.exit(2)

    mode = "simple" if args.variant == "D" else "lora"
    if args.variant == "E" and not args.lora_dir:
        print("--lora-dir is required for --variant E")
        sys.exit(2)

    output_root = ROOT / args.output_root / f"variant_{args.variant}"
    output_root.mkdir(parents=True, exist_ok=True)

    device = get_device()
    print(f"device: {device_info(device)}")
    print(f"variant: {args.variant} (mode={mode})")
    print(f"n_per_class: {args.n_per_class}")
    print(f"breeds: {breeds}")
    print(f"output: {output_root}")

    plan = []
    for breed in breeds:
        breed_dir = output_root / breed
        existing = count_existing(breed_dir, breed)
        to_generate = max(0, args.n_per_class - existing)
        plan.append((breed, breed_dir, existing, to_generate))
        print(f"  {breed}: have {existing}, need {to_generate}")

    total_to_gen = sum(n for _, _, _, n in plan)
    if total_to_gen == 0:
        print("nothing to do.")
        return
    print(f"\ntotal to generate: {total_to_gen}")

    if args.dry_run:
        return

    cfg = GenerateConfig(
        model_id=args.model_id,
        image_size=args.image_size,
        num_inference_steps=args.steps,
        guidance_scale=args.guidance,
        batch_size=args.batch_size,
    )

    pipe = None
    logger = CSVLogger(output_root, run_name=f"generate_{args.variant}_{int(time.time())}")

    total_start = time.time()
    for breed, breed_dir, existing, to_generate in plan:
        if to_generate == 0:
            continue
        if pipe is None:
            print("building pipeline...")
            pipe = build_pipeline(cfg, device)

        breed_cfg = cfg
        if args.variant == "E":
            lora_path = Path(args.lora_dir) / f"{breed}.safetensors"
            if not lora_path.exists():
                print(f"  [{breed}] SKIP - missing LoRA at {lora_path}")
                continue
            breed_cfg = GenerateConfig(
                model_id=cfg.model_id,
                image_size=cfg.image_size,
                num_inference_steps=cfg.num_inference_steps,
                guidance_scale=cfg.guidance_scale,
                batch_size=cfg.batch_size,
                lora_path=str(lora_path),
            )
            pipe.unload_lora_weights()
            pipe.load_lora_weights(str(lora_path))

        prompts, seeds = build_prompt_seed_pairs(breed, mode, to_generate, start_idx=existing)
        start = time.time()
        print(f"\n[{breed}] generating {to_generate} (idx {existing}..{existing + to_generate - 1})")
        generate_for_breed(pipe, breed, prompts, seeds, breed_dir, breed_cfg, device, logger)
        elapsed = time.time() - start
        print(f"[{breed}] done in {elapsed:.0f}s ({elapsed / to_generate:.1f}s/img)")

    total_elapsed = time.time() - total_start
    print(f"\ntotal: {total_elapsed / 60:.1f} min, log: {logger.metrics_file}")


if __name__ == "__main__":
    main()
