from pathlib import Path
from typing import List, Tuple
import re

from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF

_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
_INPUT_DIR_NAMES = ["input", "low", "lowlight", "lq"]
_TARGET_DIR_NAMES = ["target", "gt", "high", "normal", "hq"]



def _list_images(folder: Path) -> List[Path]:
    return sorted([p for p in folder.iterdir() if p.suffix.lower() in _IMAGE_EXTS and p.is_file()])


def _find_child_dir(root: Path, names: List[str]) -> Path:
    name_set = {name.lower() for name in names}
    for child in root.iterdir():
        if child.is_dir() and child.name.lower() in name_set:
            return child
    raise FileNotFoundError(f"Unable to find one of {names} under {root}")


def _pair_key(path: Path) -> str:
    stem = path.stem.lower()
    for prefix in _INPUT_DIR_NAMES + _TARGET_DIR_NAMES:
        if stem.startswith(prefix):
            stem = stem[len(prefix):]
            break
    digits = re.findall(r"\d+", stem)
    if digits:
        return digits[-1].lstrip("0") or "0"
    return stem



def _resolve_pair_dirs(root: Path) -> Tuple[Path, Path]:
    return _find_child_dir(root, _INPUT_DIR_NAMES), _find_child_dir(root, _TARGET_DIR_NAMES)


class _BasePairDataset(Dataset):
    def __init__(self, images_path, patch_size=256, mode="train"):
        self.root = Path(images_path)
        self.patch_size = int(patch_size)
        self.mode = mode
        self.input_dir, self.target_dir = _resolve_pair_dirs(self.root)

        self.inputs = _list_images(self.input_dir)
        self.targets = _list_images(self.target_dir)
        target_map = {_pair_key(p): p for p in self.targets}
        self.pairs = [(inp, target_map[_pair_key(inp)]) for inp in self.inputs if _pair_key(inp) in target_map]

        if not self.pairs:
            raise RuntimeError(f"No paired images found in {self.input_dir} and {self.target_dir}")

    def __len__(self):
        return len(self.pairs)

    def _load_pair(self, index):
        inp_path, tar_path = self.pairs[index]
        inp = Image.open(inp_path).convert("RGB")
        tar = Image.open(tar_path).convert("RGB")
        return inp, tar

    def _random_crop(self, inp, tar):
        w, h = inp.size
        if w < self.patch_size or h < self.patch_size:
            pad_w = max(0, self.patch_size - w)
            pad_h = max(0, self.patch_size - h)
            inp = TF.pad(inp, (0, 0, pad_w, pad_h))
            tar = TF.pad(tar, (0, 0, pad_w, pad_h))
            w, h = inp.size

        i, j, th, tw = torch.randint(0, h - self.patch_size + 1, (1,)).item(), torch.randint(0, w - self.patch_size + 1, (1,)).item(), self.patch_size, self.patch_size
        inp = TF.crop(inp, i, j, th, tw)
        tar = TF.crop(tar, i, j, th, tw)
        return inp, tar


class PatchDataLoaderTrain(_BasePairDataset):
    def __init__(self, images_path, options):
        super().__init__(images_path=images_path, patch_size=options.get("patch_size", 256), mode="train")

    def __getitem__(self, index):
        inp, tar = self._load_pair(index)
        inp, tar = self._random_crop(inp, tar)
        return TF.to_tensor(inp), TF.to_tensor(tar)


class PatchDataLoaderVal(_BasePairDataset):
    def __init__(self, images_path, options):
        super().__init__(images_path=images_path, patch_size=options.get("patch_size", 256), mode="val")

    def __getitem__(self, index):
        inp, tar = self._load_pair(index)
        inp, tar = self._random_crop(inp, tar)
        return TF.to_tensor(inp), TF.to_tensor(tar)


class wholeDataLoader(_BasePairDataset):
    def __init__(self, images_path, mode="train"):
        super().__init__(images_path=images_path, patch_size=0, mode=mode)

    def __getitem__(self, index):
        inp, tar = self._load_pair(index)
        return TF.to_tensor(inp), TF.to_tensor(tar)
