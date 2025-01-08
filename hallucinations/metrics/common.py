import torch


def normalize(R: torch.Tensor) -> torch.Tensor:
    "Normalizes input representations"
    with torch.no_grad():
        mean = R.mean(dim=0)
        R = R - mean
        norms = torch.norm(R, p=2, dim=1, keepdim=True)
        R = R / norms
    return R
