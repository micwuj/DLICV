import pytest
import torch

from src.models.registry import TIMM_IDS, build_model, num_trainable_params


@pytest.mark.parametrize("name", ["resnet18", "convnext_tiny", "deit_tiny"])
def test_model_builds_and_forwards(name):
    model = build_model(name, num_classes=10, pretrained=False)
    model.eval()
    x = torch.randn(2, 3, 224, 224)
    with torch.no_grad():
        out = model(x)
    assert out.shape == (2, 10)


def test_dinov2_builds_with_frozen_backbone():
    model = build_model("dinov2_small", num_classes=10, pretrained=False)
    model.eval()
    # Backbone musi byc zamrozony, head trenowalny
    backbone_trainable = sum(
        p.numel() for p in model.backbone.parameters() if p.requires_grad
    )
    head_trainable = sum(p.numel() for p in model.head.parameters() if p.requires_grad)
    assert backbone_trainable == 0
    assert head_trainable > 0

    x = torch.randn(2, 3, 224, 224)
    with torch.no_grad():
        out = model(x)
    assert out.shape == (2, 10)


def test_dinov2_train_mode_keeps_backbone_eval():
    model = build_model("dinov2_small", num_classes=10, pretrained=False)
    model.train()
    assert model.training is True
    assert model.backbone.training is False  # nadal eval
    assert model.head.training is True


def test_dinov2_head_gets_gradients_backbone_does_not():
    model = build_model("dinov2_small", num_classes=10, pretrained=False)
    model.train()
    x = torch.randn(2, 3, 224, 224)
    loss = model(x).sum()
    loss.backward()
    # Head ma grady
    assert all(p.grad is not None for p in model.head.parameters())
    # Backbone nie ma grady (requires_grad=False)
    for p in model.backbone.parameters():
        assert p.grad is None


def test_unknown_model_raises():
    with pytest.raises(ValueError, match="Nieznany model"):
        build_model("totally_not_a_model", num_classes=10, pretrained=False)


def test_num_trainable_params_reasonable():
    # ResNet-18 ma okolo 11M parametrow
    m = build_model("resnet18", num_classes=10, pretrained=False)
    n = num_trainable_params(m)
    assert 10_000_000 < n < 15_000_000


def test_all_supported_names_have_timm_id():
    for name in ("resnet18", "convnext_tiny", "deit_tiny", "dinov2_small"):
        assert name in TIMM_IDS
