"""Visual comparison of validation-selected detectors on official MVTec AD test images.

Run at the project root:
    python -m scripts.visualize_predictions --dry-run
    python -m scripts.visualize_predictions --category bottle
    python -m scripts.visualize_predictions --category tile --examples-per-case 2

Generates four-panel figures and a per-image index; never trains or tunes thresholds.
"""

import argparse
import csv
import json
import sys
from pathlib import Path

# Allow execution either as `python -m scripts.visualize_predictions` or as a file.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from src.models.autoencoder import AutoencoderDetector
from src.models.dual_unet import DualUNetDetector
from src.models.patchcore import PatchCoreDetector
from src.mvtec_dataset import MVTecAnomalyDataset

MODEL_CLASSES = {
    "autoencoder": AutoencoderDetector,
    "patchcore": PatchCoreDetector,
    "dual_unet": DualUNetDetector,
}
CASE_NAMES = {
    "tp": "verdadero_positivo",
    "fp": "falso_positivo",
    "fn": "falso_negativo",
    "tn": "verdadero_negativo",
}


def load_config():
    with (ROOT / "config.yaml").open(encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_selected_run(category, model, config):
    """Reuse the variant and fixed thresholds recorded during official evaluation."""
    result_path = (
        ROOT / config["results"]["final"] / "test" / f"{category}__{model}.json"
    )
    if not result_path.is_file():
        raise FileNotFoundError(
            f"Falta resultado oficial: {result_path}. Ejecuta primero evaluate.py"
        )
    with result_path.open(encoding="utf-8") as f:
        result = json.load(f)
    if (result["category"], result["model"]) != (category, model):
        raise ValueError(f"Inconsistencia en {result_path}")
    variant = result["selected_variant"]
    if variant not in config["models"][model]["variants"]:
        raise ValueError(f"Variante desconocida en {result_path}: {variant}")
    checkpoint = (
        ROOT / config["results"]["runs"] / model / category / variant / "model.pt"
    )
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Falta checkpoint: {checkpoint}")
    return result, checkpoint


def predict_cases(category, model, config, device):
    result, checkpoint = load_selected_run(category, model, config)
    dataset = MVTecAnomalyDataset(
        category=category,
        split="test",
        dataset_dir=ROOT / config["data"]["path"],
        splits_dir=ROOT / config["data"]["splits_path"],
        image_size=config["data"]["image_size"],
    )
    detector = MODEL_CLASSES[model].load(checkpoint, device=device)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    image_threshold = float(result["image_threshold"])
    pixel_threshold = float(result["pixel_threshold"])
    records = []

    for batch in loader:
        with torch.inference_mode():
            raw_map = detector.predict_anomaly_map(batch["image"])
        anomaly_map = raw_map.squeeze().detach().cpu().numpy().astype(np.float32)
        original = batch["image"].squeeze().permute(1, 2, 0).numpy()
        ground_truth = batch["mask"].squeeze().numpy() > 0
        label = int(batch["label"].item())
        score = float(np.max(anomaly_map))
        predicted_label = int(score >= image_threshold)
        case = (
            ("tp" if label else "fp") if predicted_label else ("fn" if label else "tn")
        )
        predicted_mask = anomaly_map >= pixel_threshold
        intersection = np.logical_and(predicted_mask, ground_truth).sum()
        denom = predicted_mask.sum() + ground_truth.sum()
        dice = float(2 * intersection / denom) if denom else 1.0
        records.append(
            {
                "path": batch["path"][0],
                "label": label,
                "predicted_label": predicted_label,
                "case": case,
                "score": score,
                "image_threshold": image_threshold,
                "pixel_threshold": pixel_threshold,
                "dice": dice,
                "image": original,
                "ground_truth": ground_truth,
                "anomaly_map": anomaly_map,
                "predicted_mask": predicted_mask,
            }
        )
    return result, records


def save_figure(record, destination, category, model, variant, vmin, vmax):
    fig, axes = plt.subplots(1, 4, figsize=(15, 4))
    axes[0].imshow(record["image"])
    axes[0].set_title("Imagen original")
    axes[1].imshow(record["ground_truth"], cmap="gray", vmin=0, vmax=1)
    axes[1].set_title("Máscara real")
    axes[2].imshow(record["anomaly_map"], cmap="inferno", vmin=vmin, vmax=vmax)
    axes[2].set_title("Mapa de anomalía")
    axes[3].imshow(record["predicted_mask"], cmap="gray", vmin=0, vmax=1)
    axes[3].set_title("Máscara predicha (umbral fijo)")
    for ax in axes:
        ax.axis("off")
    fig.suptitle(
        f"{category} | {model}/{variant} | {CASE_NAMES[record['case']]} | "
        f"score={record['score']:.4f}, Dice={record['dice']:.3f}",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(destination, dpi=160, bbox_inches="tight")
    plt.close(fig)


def visualize(category, model, config, device, examples_per_case):
    result, records = predict_cases(category, model, config, device)
    output = ROOT / config["results"]["figures"] / "test_examples" / category / model
    output.mkdir(parents=True, exist_ok=True)
    fields = [
        "category",
        "model",
        "variant",
        "path",
        "case",
        "label",
        "predicted_label",
        "score",
        "image_threshold",
        "pixel_threshold",
        "dice",
        "figure",
    ]

    # Same color scale for all figures within a given category/model.
    # Based on raw scores for visualization only; never modifies the metric or thresholds.
    all_values = np.concatenate([item["anomaly_map"].reshape(-1) for item in records])
    vmin = float(np.min(all_values))
    vmax = float(np.max(all_values))
    if vmax <= vmin:
        vmax = vmin + 1e-6

    rows = []
    for record in records:
        rows.append(
            {
                "category": category,
                "model": model,
                "variant": result["selected_variant"],
                **{
                    key: record[key]
                    for key in [
                        "path",
                        "case",
                        "label",
                        "predicted_label",
                        "score",
                        "image_threshold",
                        "pixel_threshold",
                        "dice",
                    ]
                },
                "figure": "",
            }
        )
    for case in CASE_NAMES:
        candidates = [r for r in records if r["case"] == case]
        if case == "fp":
            candidates.sort(key=lambda r: -r["score"])
        elif case == "fn":
            candidates.sort(key=lambda r: r["score"])
        elif case == "tp":
            candidates.sort(
                key=lambda r: r["dice"]
            )  # Hardest localized true positives.
        else:
            candidates.sort(
                key=lambda r: -r["score"]
            )  # Closest to an image-level false alarm.
        for index, record in enumerate(candidates[:examples_per_case], start=1):
            filename = f"{case}_{index:02d}_{Path(record['path']).stem}.png"
            save_figure(
                record,
                output / filename,
                category,
                model,
                result["selected_variant"],
                vmin,
                vmax,
            )
            for row in rows:
                if row["path"] == record["path"]:
                    row["figure"] = filename
                    break
        print(
            f"  {CASE_NAMES[case]}: {len(candidates)} imágenes; {min(len(candidates), examples_per_case)} figuras"
        )

    with (output / "image_predictions.csv").open(
        "w", newline="", encoding="utf-8"
    ) as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(f"  Guardado en: {output}")


def main():
    config = load_config()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--category", choices=config["data"]["categories"], default="bottle"
    )
    parser.add_argument(
        "--model", choices=MODEL_CLASSES, help="Omitido: los tres modelos"
    )
    parser.add_argument("--examples-per-case", type=int, default=2)
    parser.add_argument(
        "--device",
        choices=["cpu", "mps", "cuda"],
        default=config["experiment"]["device"],
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.examples_per_case < 1:
        parser.error("--examples-per-case debe ser positivo")
    models = [args.model] if args.model else list(MODEL_CLASSES)
    for model in models:
        result, checkpoint = load_selected_run(args.category, model, config)
        print(f"{args.category} / {model}: {result['selected_variant']} | {checkpoint}")
        if not args.dry_run:
            visualize(args.category, model, config, args.device, args.examples_per_case)
    if args.dry_run:
        print("Dry-run completado: no se han leído imágenes ni ejecutado inferencias.")


if __name__ == "__main__":
    main()
