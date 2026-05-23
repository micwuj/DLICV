from dataclasses import dataclass
from typing import Literal

PromptMode = Literal["simple", "lora"]


@dataclass(frozen=True)
class BreedInfo:
    key: str
    species: str
    display: str
    lora_trigger: str


BREEDS: dict[str, BreedInfo] = {
    "Maine_Coon": BreedInfo("Maine_Coon", "cat", "Maine Coon cat", "<maine_coon>"),
    "Ragdoll": BreedInfo("Ragdoll", "cat", "Ragdoll cat", "<ragdoll>"),
    "Birman": BreedInfo("Birman", "cat", "Birman cat", "<birman>"),
    "Siamese": BreedInfo("Siamese", "cat", "Siamese cat", "<siamese>"),
    "Persian": BreedInfo("Persian", "cat", "Persian cat", "<persian>"),
    "boxer": BreedInfo("boxer", "dog", "Boxer dog", "<boxer>"),
    "american_bulldog": BreedInfo("american_bulldog", "dog", "American Bulldog dog", "<american_bulldog>"),
    "staffordshire_bull_terrier": BreedInfo("staffordshire_bull_terrier", "dog", "Staffordshire Bull Terrier dog", "<staffordshire>"),
    "english_cocker_spaniel": BreedInfo("english_cocker_spaniel", "dog", "English Cocker Spaniel dog", "<english_cocker>"),
    "english_setter": BreedInfo("english_setter", "dog", "English Setter dog", "<english_setter>"),
}


POSES = (
    "sitting",
    "standing",
    "lying down",
    "looking at the camera",
    "in profile view",
    "close-up portrait shot",
)

SETTINGS = (
    "indoor",
    "outdoor in a garden",
    "on a wooden floor",
    "on grass",
    "studio lighting with plain background",
    "soft natural daylight",
)

QUALITY_TAGS = "high quality, detailed, sharp focus, realistic photograph"

NEGATIVE_PROMPT = (
    "blurry, low quality, deformed, distorted, watermark, text, signature, "
    "multiple animals, drawing, painting, cartoon, anime, render, 3d"
)


def build_prompt(
    breed_key: str,
    mode: PromptMode = "simple",
    *,
    pose: str | None = None,
    setting: str | None = None,
) -> str:
    if breed_key not in BREEDS:
        raise KeyError(f"unknown breed: {breed_key!r}")
    info = BREEDS[breed_key]
    subject = info.lora_trigger if mode == "lora" else f"a {info.display}"
    parts = [f"a photo of {subject}"]
    if pose:
        parts.append(pose)
    if setting:
        parts.append(setting)
    parts.append(QUALITY_TAGS)
    return ", ".join(parts)


def iter_prompt_variations(
    breed_key: str,
    mode: PromptMode = "simple",
) -> list[str]:
    return [
        build_prompt(breed_key, mode, pose=pose, setting=setting)
        for pose in POSES
        for setting in SETTINGS
    ]
