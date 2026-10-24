"""Carga del conjunto de validación sintética de MVTec AD."""

from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import functional as TF


class SyntheticValidationDataset(Dataset):
    """Carga imágenes normales y anomalías sintéticas previamente generadas."""

    def __init__(
        self,
        category,
        validation_dir="results/validation",
    ):
        self.category = category
        self.project_root = Path(__file__).resolve().parents[1]

        manifest_path = self.project_root / validation_dir / category / "manifest.csv"

        if not manifest_path.is_file():
            raise FileNotFoundError(f"No se encontró el manifiesto: {manifest_path}")

        self.records = pd.read_csv(
            manifest_path,
            keep_default_na=False,
        ).to_dict("records")

        if not self.records:
            raise ValueError(f"El conjunto de validación está vacío: {category}")

    def __len__(self):
        """Devuelve el número de ejemplos disponibles."""
        return len(self.records)

    def load_image(self, image_path):
        """Carga una imagen RGB como tensor [3, H, W] en [0, 1]."""
        with Image.open(image_path) as image:
            return TF.to_tensor(image.convert("RGB"))

    def load_anomaly_mask(self, mask_path, image_size):
        """Carga una máscara binaria o crea una máscara vacía."""
        if not mask_path:
            height, width = image_size
            return torch.zeros(
                (1, height, width),
                dtype=torch.float32,
            )

        with Image.open(mask_path) as mask:
            mask_tensor = TF.to_tensor(mask.convert("L"))

        return (mask_tensor > 0).float()

    def __getitem__(self, index):
        """Devuelve imagen, máscara, etiqueta y ruta."""
        record = self.records[index]

        image_path = self.project_root / record["image_path"]
        image = self.load_image(image_path)

        mask_path = record["mask_path"]

        if mask_path:
            mask_path = self.project_root / mask_path

        mask = self.load_anomaly_mask(
            mask_path,
            image_size=image.shape[-2:],
        )

        if mask.shape[-2:] != image.shape[-2:]:
            raise ValueError(f"Dimensiones de máscara incorrectas: {mask_path}")

        label = int(record["label"])

        return {
            "image": image,
            "mask": mask,
            "label": label,
            "path": str(image_path),
        }
