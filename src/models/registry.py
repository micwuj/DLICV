import timm
import torch
import torch.nn as nn

# Mapping logical name -> timm model_id
TIMM_IDS: dict[str, str] = {
    "resnet18": "resnet18.a1_in1k",
    "convnext_tiny": "convnext_tiny.fb_in22k_ft_in1k",
    "deit_tiny": "deit_tiny_patch16_224.fb_in1k",
    "dinov2_small": "vit_small_patch14_dinov2.lvd142m",
}


class DINOv2Linear(nn.Module):
    """Zamrozony backbone DINOv2 + glowica liniowa.

    Backbone jest zawsze w eval() i bez gradientow.
    Trenuje sie wylacznie head.
    """

    def __init__(self, backbone: nn.Module, num_classes: int):
        super().__init__()
        self.backbone = backbone
        for p in self.backbone.parameters():
            p.requires_grad = False
        self.backbone.eval()
        feat_dim = backbone.num_features
        self.head = nn.Linear(feat_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            feat = self.backbone(x)
        return self.head(feat)

    def train(self, mode: bool = True):
        # super().train ustawia self i wszystkie dzieci; przywracamy backbone do eval
        super().train(mode)
        self.backbone.eval()
        return self


def build_model(name: str, num_classes: int, pretrained: bool = True) -> nn.Module:
    if name not in TIMM_IDS:
        raise ValueError(
            f"Nieznany model {name!r}. Dostepne: {list(TIMM_IDS)}"
        )
    model_id = TIMM_IDS[name]

    if name == "dinov2_small":
        # DINOv2: tworzymy bez glowicy klasyfikacyjnej, na wejscie 224x224.
        # Pozycyjne embeddingi sa interpolowane z 518x518 default-u.
        backbone = timm.create_model(
            model_id, pretrained=pretrained, num_classes=0, img_size=224
        )
        return DINOv2Linear(backbone, num_classes)

    return timm.create_model(model_id, pretrained=pretrained, num_classes=num_classes)


def trainable_parameters(model: nn.Module) -> list[nn.Parameter]:
    return [p for p in model.parameters() if p.requires_grad]


def num_trainable_params(model: nn.Module) -> int:
    return sum(p.numel() for p in trainable_parameters(model))
