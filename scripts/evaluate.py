"""Evaluate validation-selected anomaly detectors on the untouched MVTec AD test set.

Run from the repository root:
    python scripts/evaluate.py --dry-run
    python scripts/evaluate.py --category bottle
    python scripts/evaluate.py

The test set is used only for evaluation: no threshold optimization or training.
"""

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.metrics import calculate_detection_metrics, calculate_segmentation_metrics
from src.models.autoencoder import AutoencoderDetector
from src.models.dual_unet import DualUNetDetector
from src.models.patchcore import PatchCoreDetector
from src.mvtec_dataset import MVTecAnomalyDataset

ROOT = Path(__file__).resolve().parents[1]
DETECTOR_CLASSES = {
    "autoencoder": AutoencoderDetector,
    "patchcore": PatchCoreDetector,
    "dual_unet": DualUNetDetector,
}


def read_configuration():
    with (ROOT / "config.yaml").open(encoding="utf-8") as source:
        return yaml.safe_load(source)


def choose_validation_variant(category, model, config):
    """Select by synthetic validation AUROC; deterministic secondary tie breaks."""
    candidates = []
    for variant in config["models"][model]["variants"]:
        directory = ROOT / config["results"]["runs"] / model / category / variant
        metrics_path = directory / "metrics.json"
        checkpoint_path = directory / "model.pt"
        if not metrics_path.is_file() or not checkpoint_path.is_file():
            raise FileNotFoundError(f"Falta checkpoint o métricas: {directory}")
        with metrics_path.open(encoding="utf-8") as source:
            metrics = json.load(source)
        if (metrics["category"], metrics["model"], metrics["variant"]) != (
            category,
            model,
            variant,
        ):
            raise ValueError(f"Identificación incompatible en {metrics_path}")
        validation = metrics["validation"]
        detection = validation["detection"]
        segmentation = validation["segmentation"]
        candidates.append(
            (
                variant,
                checkpoint_path,
                metrics,
                float(detection["image_auroc"]),
                float(segmentation["aupro"]),
            )
        )
    candidates.sort(
        key=lambda item: (-item[3], -item[4], item[2]["fit_seconds"], item[0])
    )
    return candidates[0]


def evaluate_one(category, model, config):
    variant, checkpoint_path, metrics, validation_auroc, validation_aupro = (
        choose_validation_variant(category, model, config)
    )
    image_threshold = float(metrics["validation"]["image_threshold"])
    pixel_threshold = float(metrics["validation"]["pixel_threshold"])

    dataset = MVTecAnomalyDataset(
        category=category,
        split="test",
        dataset_dir=ROOT / config["data"]["path"],
        splits_dir=ROOT / config["data"]["splits_path"],
        image_size=config["data"]["image_size"],
    )
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    device = config["experiment"]["device"]
    detector = DETECTOR_CLASSES[model].load(checkpoint_path, device=device)

    labels, scores, maps, masks = [], [], [], []
    latency_seconds = []
    for batch in loader:
        images = batch["image"]
        if device == "mps":
            torch.mps.synchronize()
        elif device == "cuda":
            torch.cuda.synchronize()
        started = time.perf_counter()
        with torch.inference_mode():
            anomaly_map = detector.predict_anomaly_map(images)
        if device == "mps":
            torch.mps.synchronize()
        elif device == "cuda":
            torch.cuda.synchronize()
        latency_seconds.append(time.perf_counter() - started)

        image_score = anomaly_map.flatten(start_dim=1).amax(dim=1)
        labels.append(int(batch["label"].item()))
        scores.append(float(image_score.item()))
        maps.append(anomaly_map.squeeze(0).squeeze(0).detach().cpu().numpy())
        masks.append(batch["mask"].squeeze(0).squeeze(0).numpy())

    detection = calculate_detection_metrics(
        np.asarray(labels),
        np.asarray(scores),
        image_threshold,
    )
    segmentation = calculate_segmentation_metrics(
        np.stack(maps),
        np.stack(masks),
        pixel_threshold,
        fpr_limit=config["evaluation"]["segmentation"]["aupro_fpr_limit"],
    )
    return {
        "category": category,
        "model": model,
        "selected_variant": variant,
        "selection_metric": "synthetic_validation_image_auroc",
        "validation_image_auroc": validation_auroc,
        "validation_aupro": validation_aupro,
        "image_threshold": image_threshold,
        "pixel_threshold": pixel_threshold,
        "test_images": len(dataset),
        "test_anomalous_images": int(sum(labels)),
        "test_normal_images": int(len(labels) - sum(labels)),
        "image_auroc": detection["image_auroc"],
        "precision": detection["precision"],
        "recall": detection["recall"],
        "f1": detection["f1"],
        "aupro": segmentation["aupro"],
        "iou": segmentation["iou"],
        "dice": segmentation["dice"],
        "latency_mean_ms": float(np.mean(latency_seconds) * 1000),
        "latency_p95_ms": float(np.percentile(latency_seconds, 95) * 1000),
        "fps": float(len(latency_seconds) / sum(latency_seconds)),
        "fit_seconds": float(metrics["fit_seconds"]),
        "confusion_matrix": detection["confusion_matrix"],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--category", choices=read_configuration()["data"]["categories"]
    )
    parser.add_argument("--model", choices=list(DETECTOR_CLASSES))
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--force", action="store_true", help="Reevaluate completed combinations"
    )
    args = parser.parse_args()

    config = read_configuration()
    categories = [args.category] if args.category else config["data"]["categories"]
    models = [args.model] if args.model else list(DETECTOR_CLASSES)
    output_directory = ROOT / config["results"]["final"] / "test"
    output_directory.mkdir(parents=True, exist_ok=True)

    for category in categories:
        for model in models:
            variant, checkpoint, _, validation_auroc, _ = choose_validation_variant(
                category, model, config
            )
            destination = output_directory / f"{category}__{model}.json"
            if destination.exists() and not args.force:
                print(f"SKIP {category} / {model}: existing result")
                continue
            print(
                f"{category} / {model} -> {variant} (validation AUROC {validation_auroc:.4f})",
                flush=True,
            )
            if args.dry_run:
                continue
            result = evaluate_one(category, model, config)
            destination.write_text(
                json.dumps(result, indent=2, allow_nan=False), encoding="utf-8"
            )
            print(
                f"  Test Image-AUROC={result['image_auroc']:.4f}, AUPRO={result['aupro']:.4f}",
                flush=True,
            )

    if args.dry_run:
        print("Dry-run finished. The test set was not read.")
        return

    all_results = []
    for file in sorted(output_directory.glob("*.json")):
        with file.open(encoding="utf-8") as source:
            all_results.append(json.load(source))
    if all_results:
        fields = [field for field in all_results[0] if field != "confusion_matrix"]
        with (output_directory / "test_summary.csv").open(
            "w", newline="", encoding="utf-8"
        ) as output:
            writer = csv.DictWriter(output, fieldnames=fields, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(all_results)
        print(f"Summary: {output_directory / 'test_summary.csv'}")


if __name__ == "__main__":
    main()
