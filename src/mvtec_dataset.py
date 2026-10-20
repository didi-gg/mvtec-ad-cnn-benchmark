"""Carga de imágenes y máscaras de MVTec AD para PyTorch."""

import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import functional as TF


class MVTecAnomalyDataset(Dataset):
    """Dataset de MVTec AD con particiones reproducibles."""

    def __init__(
        self,
        category,
        split,
        dataset_dir="data/mvtec_anomaly_detection",
        splits_dir="results/splits",
        image_size=256,
    ):
        if split not in {"train", "validation", "test"}:
            raise ValueError("split debe ser 'train', 'validation' o 'test'.")

        self.category = category
        self.split = split
        self.dataset_dir = Path(dataset_dir)
        self.image_size = image_size

        if split in {"train", "validation"}:
            manifest_path = Path(splits_dir) / f"{category}.json"

            with manifest_path.open(encoding="utf-8") as file:
                manifest = json.load(file)

            self.image_paths = [
                self.dataset_dir / relative_path for relative_path in manifest[split]
            ]

        else:
            test_dir = self.dataset_dir / category / "test"
            self.image_paths = sorted(test_dir.glob("*/*.png"))

        if not self.image_paths:
            raise ValueError(f"No se encontraron imágenes para {category}/{split}.")

    def __len__(self):
        return len(self.image_paths)

    def load_image(self, image_path):
        """Carga una imagen RGB como tensor [3, H, W] en [0, 1]."""
        with Image.open(image_path) as image:
            image = image.convert("RGB")
            image = image.resize(
                (self.image_size, self.image_size),
                Image.Resampling.BILINEAR,
            )
            return TF.to_tensor(image)

    def load_anomaly_mask(self, image_path):
        """Carga la máscara oficial; las imágenes normales usan ceros."""
        if image_path.parent.name == "good":
            return torch.zeros(
                (1, self.image_size, self.image_size),
                dtype=torch.float32,
            )

        defect_type = image_path.parent.name

        mask_path = (
            self.dataset_dir
            / self.category
            / "ground_truth"
            / defect_type
            / f"{image_path.stem}_mask.png"
        )

        if not mask_path.is_file():
            raise FileNotFoundError(f"No se encontró la máscara: {mask_path}")

        with Image.open(mask_path) as mask:
            mask = mask.convert("L")
            mask = mask.resize(
                (self.image_size, self.image_size),
                Image.Resampling.NEAREST,
            )
            mask_array = np.asarray(mask) > 0

        return torch.from_numpy(mask_array.copy()).unsqueeze(0).float()

    def __getitem__(self, index):
        image_path = self.image_paths[index]

        image = self.load_image(image_path)
        mask = self.load_anomaly_mask(image_path)

        label = int(image_path.parent.name != "good")

        return {
            "image": image,
            "mask": mask,
            "label": label,
            "path": str(image_path),
        }
