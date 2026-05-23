import json
from pathlib import Path

import pytest
import torch

from src.synth.filter import FilterConfig, nearest_sim
from src.synth.prompts import (
    BREEDS,
    NEGATIVE_PROMPT,
    build_prompt,
    iter_prompt_variations,
)

SPLIT_PATH = Path(__file__).parent.parent / "data" / "splits" / "pets_10cls_30perclass_seed0.json"


def test_breeds_match_split_classes():
    with open(SPLIT_PATH) as f:
        meta = json.load(f)["meta"]
    assert set(BREEDS.keys()) == set(meta["classes"])


def test_breed_has_required_fields():
    for key, info in BREEDS.items():
        assert info.key == key
        assert info.species in {"cat", "dog"}
        assert info.display
        assert info.lora_trigger.startswith("<") and info.lora_trigger.endswith(">")


def test_build_prompt_simple_uses_display_name():
    p = build_prompt("Ragdoll", mode="simple")
    assert "Ragdoll cat" in p
    assert "<ragdoll>" not in p


def test_build_prompt_lora_uses_trigger():
    p = build_prompt("Ragdoll", mode="lora")
    assert "<ragdoll>" in p
    assert "Ragdoll cat" not in p


def test_build_prompt_with_pose_and_setting():
    p = build_prompt("boxer", mode="simple", pose="sitting", setting="on grass")
    assert "sitting" in p and "on grass" in p


def test_build_prompt_rejects_unknown_breed():
    with pytest.raises(KeyError):
        build_prompt("doge", mode="simple")


def test_iter_prompt_variations_nonempty_and_unique():
    variations = iter_prompt_variations("Persian", mode="simple")
    assert len(variations) > 0
    assert len(set(variations)) == len(variations)


def test_negative_prompt_nonempty():
    assert isinstance(NEGATIVE_PROMPT, str) and len(NEGATIVE_PROMPT) > 10


def test_nearest_sim_topk():
    synth = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    real = torch.tensor([[1.0, 0.0], [0.9, 0.1], [0.0, 1.0]])
    synth = torch.nn.functional.normalize(synth, dim=-1)
    real = torch.nn.functional.normalize(real, dim=-1)
    sims = nearest_sim(synth, real, top_k=2)
    assert sims.shape == (2,)
    assert sims[0] > sims[1] or sims[0] >= 0.9


def test_nearest_sim_handles_empty():
    empty = torch.empty(0, 4)
    real = torch.randn(3, 4)
    real = torch.nn.functional.normalize(real, dim=-1)
    assert nearest_sim(empty, real).numel() == 0


def test_filter_config_defaults_make_sense():
    cfg = FilterConfig()
    assert 0.0 < cfg.drop_bottom_pct < 1.0
    assert 0.0 < cfg.floor_clip < 1.0
    assert 0.0 < cfg.floor_dinov2 < 1.0
    assert cfg.top_k >= 1
