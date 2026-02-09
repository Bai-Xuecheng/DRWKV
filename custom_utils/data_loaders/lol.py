from pathlib import Path
from typing import List, Tuple

from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF

_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
_INPUT_DIR_NAMES = ["input", "low", "lowlight", "lq"]
_TARGET_DIR_NAMES = ["target", "gt", "high", "normal", "hq"]



def _list_images(folder: Path) -> List[Path]:
    return sorted([p for p in folder.iterdir() if p.suffix.lower() in _IMAGE_EXTS and p.is_file()])



def _resolve_pair_dirs(root: Path) -> Tuple[Path, Path]:
    for in_name in _INPUT_DIR_NAMES:
        for gt_name in _TARGET_DIR_NAMES:
            in_dir = root / in_name
            gt_dir = root / gt_name
            if in_dir.is_dir() and gt_dir.is_dir():
                return in_dir, gt_dir
    raise FileNotFoundError(
        f"Unable to find paired folders under {root}. Expected one of {_INPUT_DIR_NAMES} and {_TARGET_DIR_NAMES}."
    )


class _BasePairDataset(Dataset):
    def __init__(self, images_path, patch_size=256, mode="train"):
        self.root = Path(images_path)
        self.patch_size = int(patch_size)
        self.mode = mode
        self.input_dir, self.target_dir = _resolve_pair_dirs(self.root)

        self.inputs = _list_images(self.input_dir)
        self.targets = _list_images(self.target_dir)
        target_map = {p.name: p for p in self.targets}
        self.pairs = [(inp, target_map[inp.name]) for inp in self.inputs if inp.name in target_map]

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

    def _center_crop(self, inp, tar):
        if self.patch_size <= 0:
            return inp, tar

        w, h = inp.size
        if w < self.patch_size or h < self.patch_size:
            pad_w = max(0, self.patch_size - w)
            pad_h = max(0, self.patch_size - h)
            inp = TF.pad(inp, (0, 0, pad_w, pad_h))
            tar = TF.pad(tar, (0, 0, pad_w, pad_h))
            w, h = inp.size

        i = (h - self.patch_size) // 2
        j = (w - self.patch_size) // 2
        inp = TF.crop(inp, i, j, self.patch_size, self.patch_size)
        tar = TF.crop(tar, i, j, self.patch_size, self.patch_size)
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
        inp, tar = self._center_crop(inp, tar)
        return TF.to_tensor(inp), TF.to_tensor(tar)


class wholeDataLoader(_BasePairDataset):
    def __init__(self, images_path, mode="train"):
        super().__init__(images_path=images_path, patch_size=0, mode=mode)

    def __getitem__(self, index):
        inp, tar = self._load_pair(index)
        return TF.to_tensor(inp), TF.to_tensor(tar)
