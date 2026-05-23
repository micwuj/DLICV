import pytest
import torch
from PIL import Image

from src.data.transforms import make_eval_transform, make_train_transform


def _dummy_img():
    return Image.new("RGB", (300, 300), color="blue")


def test_eval_transform_shape_and_range():
    tf = make_eval_transform(224)
    out = tf(_dummy_img())
    assert isinstance(out, torch.Tensor)
    assert out.shape == (3, 224, 224)
    assert out.dtype == torch.float32
    # Po normalizacji ImageNet wartosci powinny byc w rozsadnym zakresie
    assert out.abs().max().item() < 5


def test_train_transform_none_aug():
    tf = make_train_transform(224, augmentation="none")
    out = tf(_dummy_img())
    assert out.shape == (3, 224, 224)


def test_train_transform_classical_aug():
    tf = make_train_transform(224, augmentation="classical")
    out = tf(_dummy_img())
    assert out.shape == (3, 224, 224)


def test_train_transform_unknown_aug_raises():
    with pytest.raises(ValueError, match="augmentation"):
        make_train_transform(224, augmentation="foobar")


def test_eval_transform_different_size():
    tf = make_eval_transform(96)
    out = tf(_dummy_img())
    assert out.shape == (3, 96, 96)
