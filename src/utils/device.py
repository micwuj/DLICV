import torch


def get_device(prefer: str = "auto") -> torch.device:
    if prefer != "auto":
        return torch.device(prefer)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def device_info(device: torch.device | None = None) -> str:
    d = device if device is not None else get_device()
    if d.type == "cuda":
        name = torch.cuda.get_device_name(0)
        mem_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        return f"cuda ({name}, {mem_gb:.1f} GB)"
    if d.type == "mps":
        return "mps (Apple Silicon)"
    return "cpu"
