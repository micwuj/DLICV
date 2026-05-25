"""Minimal LoRA fine-tuning of SD 1.5 UNet for one breed.

Trains attention LoRA adapters (q/k/v/o projections) on a handful of real
images of a single breed, using the trigger token from src/synth/prompts.py.

Example:
    python scripts/train_lora.py \\
        --breed Ragdoll \\
        --train-data-dir data/lora_train/Ragdoll \\
        --output-dir data/loras/Ragdoll \\
        --steps 500 --rank 16
"""
import argparse
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.synth.prompts import BREEDS
from src.utils.device import device_info, get_device
from src.utils.seed import set_seed


def list_images(d: Path) -> list[Path]:
    return sorted([p for p in d.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}])


def preprocess_image(path: Path, size: int) -> torch.Tensor:
    img = Image.open(path).convert("RGB")
    w, h = img.size
    s = min(w, h)
    left, top = (w - s) // 2, (h - s) // 2
    img = img.crop((left, top, left + s, top + s)).resize((size, size), Image.LANCZOS)
    import numpy as np
    arr = np.array(img).astype("float32") / 127.5 - 1.0
    return torch.from_numpy(arr).permute(2, 0, 1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--breed", required=True)
    ap.add_argument("--train-data-dir", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--model-id", default="runwayml/stable-diffusion-v1-5")
    ap.add_argument("--steps", type=int, default=500)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--resolution", type=int, default=512)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--instance-prompt", default=None,
                    help="Override; default uses trigger token from prompts.py")
    args = ap.parse_args()

    if args.breed not in BREEDS:
        print(f"unknown breed: {args.breed}; valid: {list(BREEDS.keys())}")
        sys.exit(2)
    info = BREEDS[args.breed]
    prompt = args.instance_prompt or f"a photo of {info.lora_trigger} {info.species}"

    set_seed(args.seed)
    device = get_device()
    print(f"device: {device_info(device)}")
    print(f"breed: {args.breed} | prompt: {prompt}")

    train_dir = Path(args.train_data_dir)
    images = list_images(train_dir)
    if not images:
        print(f"no images in {train_dir}")
        sys.exit(2)
    print(f"training on {len(images)} images")

    from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel
    from peft import LoraConfig, get_peft_model_state_dict
    from safetensors.torch import save_file
    from transformers import CLIPTextModel, CLIPTokenizer

    dtype = torch.float16 if device.type == "cuda" else torch.float32

    print("loading components...")
    tokenizer = CLIPTokenizer.from_pretrained(args.model_id, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(args.model_id, subfolder="text_encoder", torch_dtype=dtype).to(device).eval()
    vae = AutoencoderKL.from_pretrained(args.model_id, subfolder="vae", torch_dtype=dtype).to(device).eval()
    unet = UNet2DConditionModel.from_pretrained(args.model_id, subfolder="unet", torch_dtype=torch.float32).to(device)
    noise_scheduler = DDPMScheduler.from_pretrained(args.model_id, subfolder="scheduler")

    for p in text_encoder.parameters():
        p.requires_grad_(False)
    for p in vae.parameters():
        p.requires_grad_(False)
    for p in unet.parameters():
        p.requires_grad_(False)

    lora_config = LoraConfig(
        r=args.rank,
        lora_alpha=args.rank,
        target_modules=["to_q", "to_k", "to_v", "to_out.0"],
        init_lora_weights="gaussian",
    )
    unet.add_adapter(lora_config)
    trainable = [p for p in unet.parameters() if p.requires_grad]
    n_trainable = sum(p.numel() for p in trainable)
    print(f"LoRA trainable params: {n_trainable:,}")

    optimizer = torch.optim.AdamW(trainable, lr=args.lr)

    with torch.no_grad():
        tokens = tokenizer(
            [prompt],
            padding="max_length",
            max_length=tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        ).input_ids.to(device)
        encoder_hidden_states = text_encoder(tokens)[0]

    pixel_tensors = [preprocess_image(p, args.resolution) for p in images]

    unet.train()
    rng = torch.Generator(device="cpu").manual_seed(args.seed)
    for step in range(args.steps):
        idx = int(torch.randint(0, len(pixel_tensors), (1,), generator=rng).item())
        pixel = pixel_tensors[idx].unsqueeze(0).to(device, dtype=dtype)

        with torch.no_grad():
            latents = vae.encode(pixel).latent_dist.sample() * vae.config.scaling_factor
        latents = latents.to(torch.float32)

        noise = torch.randn_like(latents)
        timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (1,), device=device).long()
        noisy = noise_scheduler.add_noise(latents, noise, timesteps)

        pred = unet(noisy, timesteps, encoder_hidden_states=encoder_hidden_states.to(torch.float32)).sample
        loss = F.mse_loss(pred, noise)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        if step % 50 == 0 or step == args.steps - 1:
            print(f"  step {step:4d}/{args.steps}  loss={loss.item():.4f}")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    lora_state = get_peft_model_state_dict(unet)
    out_path = out_dir / f"{args.breed}.safetensors"
    save_file(lora_state, out_path)
    print(f"saved -> {out_path} ({out_path.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
