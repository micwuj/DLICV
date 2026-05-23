import torch

from src.utils.config import deep_merge, load_with_overrides
from src.utils.device import device_info, get_device
from src.utils.logger import CSVLogger
from src.utils.seed import set_seed


def test_device_resolves():
    d = get_device()
    assert d.type in {"cuda", "mps", "cpu"}
    assert isinstance(device_info(d), str)


def test_device_force_cpu():
    assert get_device("cpu").type == "cpu"


def test_seed_is_reproducible():
    set_seed(42)
    a = torch.rand(5)
    set_seed(42)
    b = torch.rand(5)
    assert torch.allclose(a, b)


def test_deep_merge_overrides_leaf():
    base = {"a": {"b": 1, "c": 2}}
    over = {"a": {"b": 99}}
    merged = deep_merge(base, over)
    assert merged == {"a": {"b": 99, "c": 2}}


def test_load_with_overrides(tmp_path):
    base = tmp_path / "base.yaml"
    over = tmp_path / "over.yaml"
    base.write_text("training:\n  lr: 0.001\n  bs: 32\n")
    over.write_text("training:\n  lr: 0.01\n")
    cfg = load_with_overrides(base, over)
    assert cfg["training"]["lr"] == 0.01
    assert cfg["training"]["bs"] == 32


def test_csv_logger_writes_header_and_rows(tmp_path):
    logger = CSVLogger(tmp_path, run_name="t")
    logger.log(epoch=1, loss=0.5)
    logger.log(epoch=2, loss=0.3)
    lines = (tmp_path / "t_metrics.csv").read_text().strip().split("\n")
    assert lines[0] == "epoch,loss"
    assert len(lines) == 3


def test_shipped_base_config_parses():
    cfg = load_with_overrides("configs/base.yaml")
    assert cfg["data"]["num_classes"] == 10
    assert len(cfg["data"]["classes"]) == 10
    assert cfg["training"]["batch_size"] > 0


def test_shipped_cpu_smoke_override_applies():
    cfg = load_with_overrides("configs/base.yaml", "configs/cpu_smoke.yaml")
    assert cfg["device"] == "cpu"
    assert cfg["training"]["batch_size"] == 4
    # Inherited from base:
    assert cfg["data"]["num_classes"] == 10
