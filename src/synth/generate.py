from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
from PIL import Image

from src.synth.prompts import NEGATIVE_PROMPT
from src.utils.device import get_device
from src.utils.logger import CSVLogger


@dataclass
class GenerateConfig:
    model_id: str = "runwayml/stable-diffusion-v1-5"
    image_size: int = 512
    num_inference_steps: int = 30
    guidance_scale: float = 7.5
    batch_size: int = 4
    lora_path: str | None = None
    negative_prompt: str = NEGATIVE_PROMPT
    extra: dict[str, Any] = field(default_factory=dict)


def build_pipeline(cfg: GenerateConfig, device: torch.device | None = None):
    from diffusers import StableDiffusionPipeline

    device = device or get_device()
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    pipe = StableDiffusionPipeline.from_pretrained(cfg.model_id, torch_dtype=dtype, safety_checker=None)
    pipe = pipe.to(device)
    if cfg.lora_path is not None:
        pipe.load_lora_weights(cfg.lora_path)
    pipe.set_progress_bar_config(disable=True)
    return pipe


def _chunk(seq: list, n: int):
    for i in range(0, len(seq), n):
        yield seq[i : i + n]


def generate_batch(
    pipe,
    prompts: list[str],
    seeds: list[int],
    cfg: GenerateConfig,
    device: torch.device,
) -> list[Image.Image]:
    if len(prompts) != len(seeds):
        raise ValueError("prompts and seeds must be the same length")
    generators = [torch.Generator(device=device.type).manual_seed(s) for s in seeds]
    result = pipe(
        prompt=prompts,
        negative_prompt=[cfg.negative_prompt] * len(prompts),
        num_inference_steps=cfg.num_inference_steps,
        guidance_scale=cfg.guidance_scale,
        width=cfg.image_size,
        height=cfg.image_size,
        generator=generators,
    )
    return result.images


def generate_for_breed(
    pipe,
    breed_key: str,
    prompts: list[str],
    seeds: list[int],
    output_dir: Path,
    cfg: GenerateConfig,
    device: torch.device,
    logger: CSVLogger | None = None,
) -> list[Path]:
    if len(prompts) != len(seeds):
        raise ValueError("prompts and seeds must be the same length")
    output_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    idx = 0
    for batch_prompts, batch_seeds in zip(_chunk(prompts, cfg.batch_size), _chunk(seeds, cfg.batch_size)):
        images = generate_batch(pipe, batch_prompts, batch_seeds, cfg, device)
        for img, prompt, seed in zip(images, batch_prompts, batch_seeds):
            out_path = output_dir / f"{breed_key}_{idx:04d}.png"
            img.save(out_path)
            paths.append(out_path)
            if logger is not None:
                logger.log(
                    breed=breed_key,
                    idx=idx,
                    seed=seed,
                    prompt=prompt,
                    image_size=cfg.image_size,
                    steps=cfg.num_inference_steps,
                    guidance=cfg.guidance_scale,
                    lora=cfg.lora_path or "",
                    path=str(out_path),
                )
            idx += 1
    return paths
