from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import torch
import torch.nn.functional as F
from PIL import Image

from src.utils.device import get_device

EmbedKind = Literal["clip", "dinov2"]


@dataclass
class FilterConfig:
    clip_model_id: str = "openai/clip-vit-base-patch32"
    dinov2_model_id: str = "facebook/dinov2-small"
    batch_size: int = 16
    top_k: int = 5
    drop_bottom_pct: float = 0.25
    floor_clip: float = 0.65
    floor_dinov2: float = 0.55


@dataclass
class FilterResult:
    kept: list[Path]
    dropped: list[Path]
    clip_sim: list[float]
    dinov2_sim: list[float]
    reasons: list[str]


def _load_embedder(kind: EmbedKind, model_id: str, device: torch.device):
    from transformers import AutoImageProcessor, AutoModel, CLIPModel, CLIPProcessor

    if kind == "clip":
        model = CLIPModel.from_pretrained(model_id).to(device).eval()
        processor = CLIPProcessor.from_pretrained(model_id)
        return model, processor
    if kind == "dinov2":
        model = AutoModel.from_pretrained(model_id).to(device).eval()
        processor = AutoImageProcessor.from_pretrained(model_id)
        return model, processor
    raise ValueError(f"unknown kind: {kind}")


@torch.no_grad()
def compute_embeddings(
    image_paths: list[Path],
    kind: EmbedKind,
    model_id: str,
    device: torch.device,
    batch_size: int = 16,
) -> torch.Tensor:
    model, processor = _load_embedder(kind, model_id, device)
    feats: list[torch.Tensor] = []
    for i in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[i : i + batch_size]
        images = [Image.open(p).convert("RGB") for p in batch_paths]
        if kind == "clip":
            inputs = processor(images=images, return_tensors="pt").to(device)
            out = model.get_image_features(**inputs)
        else:
            inputs = processor(images=images, return_tensors="pt").to(device)
            out = model(**inputs).last_hidden_state[:, 0]
        out = F.normalize(out, dim=-1)
        feats.append(out.cpu())
    return torch.cat(feats, dim=0) if feats else torch.empty(0)


def nearest_sim(synth_emb: torch.Tensor, real_emb: torch.Tensor, top_k: int = 5) -> torch.Tensor:
    if synth_emb.numel() == 0 or real_emb.numel() == 0:
        return torch.zeros(synth_emb.shape[0])
    sim = synth_emb @ real_emb.T
    k = min(top_k, real_emb.shape[0])
    topk, _ = sim.topk(k=k, dim=1)
    return topk.mean(dim=1)


def filter_synthetic(
    synthetic_paths: list[Path],
    real_ref_paths: list[Path],
    cfg: FilterConfig,
    device: torch.device | None = None,
) -> FilterResult:
    device = device or get_device()
    synth_clip = compute_embeddings(synthetic_paths, "clip", cfg.clip_model_id, device, cfg.batch_size)
    real_clip = compute_embeddings(real_ref_paths, "clip", cfg.clip_model_id, device, cfg.batch_size)
    synth_dino = compute_embeddings(synthetic_paths, "dinov2", cfg.dinov2_model_id, device, cfg.batch_size)
    real_dino = compute_embeddings(real_ref_paths, "dinov2", cfg.dinov2_model_id, device, cfg.batch_size)

    clip_sim = nearest_sim(synth_clip, real_clip, cfg.top_k)
    dino_sim = nearest_sim(synth_dino, real_dino, cfg.top_k)

    # Percentile threshold is computed jointly via a combined score so that an
    # image can compensate weakness in one space with strength in the other,
    # but absolute floors are enforced per metric.
    combined = 0.5 * clip_sim + 0.5 * dino_sim
    if combined.numel() > 0:
        pct_threshold = torch.quantile(combined, cfg.drop_bottom_pct).item()
    else:
        pct_threshold = float("-inf")

    kept: list[Path] = []
    dropped: list[Path] = []
    reasons: list[str] = []
    for i, p in enumerate(synthetic_paths):
        c = clip_sim[i].item()
        d = dino_sim[i].item()
        s = combined[i].item()
        why = []
        if c < cfg.floor_clip:
            why.append(f"clip<{cfg.floor_clip}")
        if d < cfg.floor_dinov2:
            why.append(f"dinov2<{cfg.floor_dinov2}")
        if s < pct_threshold:
            why.append(f"below_pct{cfg.drop_bottom_pct}")
        if why:
            dropped.append(p)
            reasons.append("|".join(why))
        else:
            kept.append(p)
            reasons.append("")

    return FilterResult(
        kept=kept,
        dropped=dropped,
        clip_sim=clip_sim.tolist(),
        dinov2_sim=dino_sim.tolist(),
        reasons=reasons,
    )
