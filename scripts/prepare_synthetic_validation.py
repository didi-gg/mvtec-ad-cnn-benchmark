"""Genera un conjunto de validación sintética común para los tres modelos.

Uso:
    python scripts/prepare_synthetic_validation.py --category bottle
    python scripts/prepare_synthetic_validation.py --all

Guarda imágenes, máscaras y un manifiesto CSV en results/validation/.
"""

import argparse
import csv
import hashlib
from pathlib import Path

from PIL import Image
from tqdm import tqdm

from src.mvtec_dataset import MVTecAnomalyDataset
from src.synthetic_anomalies import generate_synthetic_anomaly

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = PROJECT_ROOT / "results" / "validation"
SPLITS_ROOT = PROJECT_ROOT / "results" / "splits"

SEED = 2026
IMAGE_SIZE = 256

ANOMALY_METHODS = ("cutpaste", "irregular_texture")


def create_image_seed(category, image_path, method):
    """Obtiene una semilla estable para cada imagen y método."""
    identifier = f"{SEED}:{category}:{image_path}:{method}"
    digest = hashlib.sha256(identifier.encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "big")


def save_rgb_image(image, destination):
    """Guarda un tensor RGB en formato PNG."""
    array = image.clamp(0, 1).permute(1, 2, 0).mul(255).byte().numpy()

    Image.fromarray(array).save(destination)


def save_binary_mask(mask, destination):
    """Guarda una máscara binaria con valores 0 y 255."""
    array = mask.squeeze(0).mul(255).byte().numpy()
    Image.fromarray(array, mode="L").save(destination)


def prepare_category(category):
    """Genera anomalías sintéticas de validación para una categoría."""
    dataset = MVTecAnomalyDataset(
        category=category,
        split="validation",
        image_size=IMAGE_SIZE,
    )

    category_output = OUTPUT_ROOT / category
    images_dir = category_output / "images"
    masks_dir = category_output / "masks"

    images_dir.mkdir(parents=True, exist_ok=True)
    masks_dir.mkdir(parents=True, exist_ok=True)

    records = []

    for sample in tqdm(dataset, desc=category):
        original_image = sample["image"]
        original_path = Path(sample["path"])
        image_id = original_path.stem

        # Guardar también la imagen normal redimensionada.
        normal_name = f"{image_id}_normal.png"
        normal_path = images_dir / normal_name

        save_rgb_image(original_image, normal_path)

        records.append(
            {
                "category": category,
                "source_image": str(original_path),
                "image_path": str(normal_path.relative_to(PROJECT_ROOT)),
                "mask_path": "",
                "label": 0,
                "method": "normal",
                "seed": "",
            }
        )

        for method in ANOMALY_METHODS:
            seed = create_image_seed(category, original_path, method)

            modified_image, mask = generate_synthetic_anomaly(
                original_image,
                method=method,
                seed=seed,
            )

            if mask.sum().item() == 0:
                raise RuntimeError(f"Máscara vacía: {category}/{image_id}/{method}")

            modified_name = f"{image_id}_{method}.png"
            mask_name = f"{image_id}_{method}_mask.png"

            modified_path = images_dir / modified_name
            mask_path = masks_dir / mask_name

            save_rgb_image(modified_image, modified_path)
            save_binary_mask(mask, mask_path)

            records.append(
                {
                    "category": category,
                    "source_image": str(original_path),
                    "image_path": str(modified_path.relative_to(PROJECT_ROOT)),
                    "mask_path": str(mask_path.relative_to(PROJECT_ROOT)),
                    "label": 1,
                    "method": method,
                    "seed": seed,
                }
            )

    manifest_path = category_output / "manifest.csv"

    with manifest_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "category",
                "source_image",
                "image_path",
                "mask_path",
                "label",
                "method",
                "seed",
            ],
        )
        writer.writeheader()
        writer.writerows(records)

    print(
        f"{category}: {len(dataset)} normales, "
        f"{len(dataset) * len(ANOMALY_METHODS)} anomalías, "
        f"{len(records)} ejemplos en total."
    )

    print(f"Manifiesto: {manifest_path}")


def main():
    parser = argparse.ArgumentParser()

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--category", type=str)
    group.add_argument("--all", action="store_true")

    args = parser.parse_args()

    if args.all:
        categories = sorted(path.stem for path in SPLITS_ROOT.glob("*.json"))
    else:
        categories = [args.category]

    for category in categories:
        prepare_category(category)


if __name__ == "__main__":
    main()
