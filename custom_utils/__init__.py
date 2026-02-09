import os
import glob
from pathlib import Path

import torch
import torch.nn.functional as F


def mkdir(path):
    os.makedirs(path, exist_ok=True)


def get_last_path(path, session):
    files = sorted(glob.glob(os.path.join(path, f"*{session}")))
    if not files:
        raise FileNotFoundError(f"No checkpoint ending with '{session}' found in {path}")
    return files[-1]


def load_checkpoint(model, weights):
    checkpoint = torch.load(weights, map_location="cpu")
    state_dict = checkpoint.get("state_dict", checkpoint)
    model.load_state_dict(state_dict, strict=True)


def load_start_epoch(weights):
    checkpoint = torch.load(weights, map_location="cpu")
    return checkpoint.get("epoch", 0)


def load_optim(optimizer, weights):
    checkpoint = torch.load(weights, map_location="cpu")
    if "optimizer" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])


def _to_4d(img):
    if img.ndim == 3:
        return img.unsqueeze(0)
    return img


def torchPSNR(pred, target, max_val=1.0):
    pred = torch.clamp(pred, 0, max_val)
    target = torch.clamp(target, 0, max_val)
    mse = F.mse_loss(pred, target)
    if mse <= 0:
        return torch.tensor(float("inf"), device=pred.device)
    return 20 * torch.log10(torch.tensor(max_val, device=pred.device)) - 10 * torch.log10(mse)


def torchSSIM(pred, target, window_size=11):
    pred = _to_4d(pred)
    target = _to_4d(target)
    c1 = 0.01 ** 2
    c2 = 0.03 ** 2

    mu_x = F.avg_pool2d(pred, window_size, stride=1, padding=window_size // 2)
    mu_y = F.avg_pool2d(target, window_size, stride=1, padding=window_size // 2)

    sigma_x = F.avg_pool2d(pred * pred, window_size, stride=1, padding=window_size // 2) - mu_x ** 2
    sigma_y = F.avg_pool2d(target * target, window_size, stride=1, padding=window_size // 2) - mu_y ** 2
    sigma_xy = F.avg_pool2d(pred * target, window_size, stride=1, padding=window_size // 2) - mu_x * mu_y

    ssim_map = ((2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)) / ((mu_x ** 2 + mu_y ** 2 + c1) * (sigma_x + sigma_y + c2))
    return ssim_map.mean()


def network_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


__all__ = [
    "mkdir",
    "get_last_path",
    "load_checkpoint",
    "load_start_epoch",
    "load_optim",
    "torchPSNR",
    "torchSSIM",
    "network_parameters",
]
