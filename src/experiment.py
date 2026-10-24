"""Ejecución y validación de experimentos con MVTec AD.

Ejemplos:
    python -m src.experiment --category bottle --model autoencoder
    python -m src.experiment --category bottle --model patchcore
    python -m src.experiment --category bottle --model dual_unet

Cada ejecución:
- Utiliza únicamente imágenes normales para ajustar los detectores.
- Compara las dos variantes del modelo.
- Selecciona checkpoints mediante validación sintética.
- Calibra umbrales sin utilizar el test oficial.
- Guarda checkpoints y métricas reproducibles.
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader

from src.metrics import (
    calculate_detection_metrics,
    calculate_segmentation_metrics,
    find_best_image_threshold,
    find_best_pixel_threshold,
)
from src.models.autoencoder import AutoencoderDetector
from src.models.dual_unet import DualUNetDetector
from src.models.patchcore import PatchCoreDetector
from src.mvtec_dataset import MVTecAnomalyDataset
from src.synthetic_validation_dataset import SyntheticValidationDataset

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def synchronize_device(device):
    """Espera a que terminen las operaciones de GPU."""
    device = torch.device(device)

    if device.type == "mps":
        torch.mps.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize()


def load_experiment_config():
    """Lee la configuración general del proyecto."""
    config_path = PROJECT_ROOT / "config.yaml"

    with config_path.open(encoding="utf-8") as file:
        return yaml.safe_load(file)


def set_experiment_seed(seed):
    """Establece las semillas de NumPy y PyTorch."""
    np.random.seed(seed)
    torch.manual_seed(seed)


def create_detector(model_name, variant_name, config):
    """Construye una variante de uno de los tres detectores."""
    settings = config["models"][model_name]
    variant = settings["variants"][variant_name]

    device = config["experiment"]["device"]
    seed = config["project"]["seed"]
    max_epochs = config["experiment"]["max_epochs"]

    if model_name == "autoencoder":
        return AutoencoderDetector(
            loss_name=variant["loss"],
            channels=settings["channels"],
            learning_rate=settings["learning_rate"],
            epochs=max_epochs,
            device=device,
        )

    if model_name == "patchcore":
        return PatchCoreDetector(
            memory_fraction=variant["fraction"],
            device=device,
            seed=seed,
            pretrained=settings["pretrained"],
        )

    if model_name == "dual_unet":
        return DualUNetDetector(
            channels=variant["channels"],
            learning_rate=settings["learning_rate"],
            epochs=max_epochs,
            device=device,
            seed=seed,
        )

    raise ValueError(f"Modelo desconocido: {model_name}")


def create_data_loaders(category, model_name, config):
    """Prepara entrenamiento normal y validación sintética común."""
    settings = config["models"][model_name]

    training_dataset = MVTecAnomalyDataset(
        category=category,
        split="train",
        dataset_dir=PROJECT_ROOT / config["data"]["path"],
        splits_dir=PROJECT_ROOT / config["data"]["splits_path"],
        image_size=config["data"]["image_size"],
    )

    validation_dataset = SyntheticValidationDataset(
        category=category,
        validation_dir=config["validation"]["output_path"],
    )

    # La semilla se reinicia para cada variante.
    # Así las variantes de un mismo modelo reciben el mismo
    # orden de imágenes de entrenamiento.
    generator = torch.Generator().manual_seed(config["project"]["seed"])

    training_loader = DataLoader(
        training_dataset,
        batch_size=settings["batch_size"],
        shuffle=True,
        generator=generator,
        num_workers=0,
    )

    validation_loader = DataLoader(
        validation_dataset,
        batch_size=4,
        shuffle=False,
        num_workers=0,
    )

    return training_loader, validation_loader


def create_optimizer(detector, model_name):
    """Crea el optimizador de las redes entrenables."""
    if model_name == "autoencoder":
        parameters = detector.network.parameters()

    elif model_name == "dual_unet":
        parameters = list(detector.reconstruction_network.parameters()) + list(
            detector.segmentation_network.parameters()
        )

    else:
        raise ValueError("PatchCore no requiere un optimizador de entrenamiento.")

    return torch.optim.Adam(
        parameters,
        lr=detector.learning_rate,
    )


def train_one_epoch(detector, model_name, training_loader, optimizer, rng):
    """Entrena una época de Autoencoder o Dual U-Net."""
    if model_name == "autoencoder":
        detector.network.train()
    else:
        detector.reconstruction_network.train()
        detector.segmentation_network.train()

    total_loss = 0.0
    total_images = 0

    for batch in training_loader:
        normal_images = batch["image"]

        if torch.any(batch["label"] != 0):
            raise ValueError("El entrenamiento solo admite imágenes normales.")

        optimizer.zero_grad(set_to_none=True)

        if model_name == "autoencoder":
            images = normal_images.to(detector.device)

            reconstructions = detector.network(images)

            loss = detector.compute_reconstruction_loss(
                images,
                reconstructions,
            )

        else:
            modified_images, anomaly_masks = detector.generate_training_anomalies(
                normal_images,
                rng,
            )

            normal_images = normal_images.to(detector.device)
            modified_images = modified_images.to(detector.device)
            anomaly_masks = anomaly_masks.to(detector.device)

            reconstructed_images = detector.reconstruct_images(modified_images)

            segmentation_logits = detector.segmentation_network(
                torch.cat(
                    [modified_images, reconstructed_images],
                    dim=1,
                )
            )

            reconstruction_loss = F.l1_loss(
                reconstructed_images,
                normal_images,
            )

            segmentation_loss = detector.compute_segmentation_loss(
                segmentation_logits,
                anomaly_masks,
            )

            loss = reconstruction_loss + segmentation_loss

        loss.backward()
        optimizer.step()

        batch_size = normal_images.shape[0]

        total_loss += loss.item() * batch_size
        total_images += batch_size

    synchronize_device(detector.device)

    if total_images == 0:
        raise ValueError("El conjunto de entrenamiento está vacío.")

    return total_loss / total_images


def predict_validation(detector, validation_loader):
    """Obtiene las predicciones del conjunto sintético común."""
    image_labels = []
    image_scores = []
    anomaly_maps = []
    anomaly_masks = []

    for batch in validation_loader:
        predicted_maps = detector.predict_anomaly_map(batch["image"])

        # Regla común definida en AnomalyDetector.
        predicted_scores = predicted_maps.flatten(start_dim=1).amax(dim=1)

        image_labels.append(batch["label"].numpy())

        image_scores.append(predicted_scores.detach().cpu().numpy())

        anomaly_maps.append(predicted_maps.detach().cpu().numpy())

        anomaly_masks.append(batch["mask"].numpy())

    return {
        "labels": np.concatenate(image_labels),
        "scores": np.concatenate(image_scores),
        "anomaly_maps": np.concatenate(anomaly_maps),
        "masks": np.concatenate(anomaly_masks),
    }


def calculate_validation_auroc(detector, validation_loader):
    """Calcula Image-AUROC para seleccionar checkpoints."""
    predictions = predict_validation(
        detector,
        validation_loader,
    )

    return float(
        roc_auc_score(
            predictions["labels"],
            predictions["scores"],
        )
    )


def train_detector(
    detector,
    model_name,
    training_loader,
    validation_loader,
    config,
    deadline,
    checkpoint_path,
):
    """Entrena y conserva el checkpoint con mejor Image-AUROC."""
    optimizer = create_optimizer(detector, model_name)

    rng = np.random.default_rng(config["project"]["seed"])

    max_epochs = config["experiment"]["max_epochs"]
    patience = config["models"][model_name]["patience"]

    validation_interval = 5

    best_auroc = -1.0
    best_epoch = None
    checks_without_improvement = 0

    history = []
    stop_reason = "max_epochs"

    for epoch in range(1, max_epochs + 1):
        if time.perf_counter() >= deadline:
            stop_reason = "time_budget"
            print("Presupuesto temporal alcanzado.")
            break

        synchronize_device(detector.device)
        epoch_start = time.perf_counter()

        training_loss = train_one_epoch(
            detector,
            model_name,
            training_loader,
            optimizer,
            rng,
        )

        epoch_seconds = time.perf_counter() - epoch_start

        record = {
            "epoch": epoch,
            "training_loss": float(training_loss),
            "epoch_seconds": round(epoch_seconds, 3),
        }

        should_validate = (
            epoch == 1
            or epoch % validation_interval == 0
            or epoch == max_epochs
            or time.perf_counter() >= deadline
        )

        if should_validate:
            validation_auroc = calculate_validation_auroc(
                detector,
                validation_loader,
            )

            record["validation_image_auroc"] = validation_auroc

            if validation_auroc > best_auroc:
                best_auroc = validation_auroc
                best_epoch = epoch
                checks_without_improvement = 0

                # Se guarda el mejor modelo, no el último.
                detector.save(checkpoint_path)

            else:
                checks_without_improvement += 1

            print(
                f"Época {epoch:3d} | "
                f"Loss {training_loss:.5f} | "
                f"Val AUROC {validation_auroc:.4f}",
                flush=True,
            )

            if checks_without_improvement >= patience:
                stop_reason = "early_stopping"
                print("Early stopping activado.")
                history.append(record)
                break

        history.append(record)

        # El presupuesto se comprueba entre épocas.
        if time.perf_counter() >= deadline:
            stop_reason = "time_budget"
            print("Presupuesto temporal alcanzado.")
            break

    if best_epoch is None:
        raise RuntimeError("No se pudo seleccionar un checkpoint de validación.")

    best_detector = type(detector).load(
        checkpoint_path,
        device=str(detector.device),
    )

    return {
        "detector": best_detector,
        "history": history,
        "best_epoch": best_epoch,
        "best_validation_image_auroc": best_auroc,
        "stop_reason": stop_reason,
    }


def evaluate_validation(detector, validation_loader, config):
    """Selecciona umbrales y calcula métricas de validación."""
    predictions = predict_validation(
        detector,
        validation_loader,
    )

    image_threshold = find_best_image_threshold(
        predictions["labels"],
        predictions["scores"],
    )

    pixel_threshold = find_best_pixel_threshold(
        predictions["anomaly_maps"],
        predictions["masks"],
    )

    detection_metrics = calculate_detection_metrics(
        predictions["labels"],
        predictions["scores"],
        image_threshold,
    )

    segmentation_metrics = calculate_segmentation_metrics(
        predictions["anomaly_maps"],
        predictions["masks"],
        pixel_threshold,
        fpr_limit=config["evaluation"]["segmentation"]["aupro_fpr_limit"],
    )

    return {
        "image_threshold": image_threshold,
        "pixel_threshold": pixel_threshold,
        "detection": detection_metrics,
        "segmentation": segmentation_metrics,
    }


def run_experiment(category, model_name, config):
    """Ejecuta las dos variantes de un modelo en una categoría."""
    if category not in config["data"]["categories"]:
        raise ValueError(f"Categoría desconocida: {category}")

    if model_name not in config["models"]:
        raise ValueError(f"Modelo desconocido: {model_name}")

    settings = config["models"][model_name]
    variants = settings["variants"]

    budget_minutes = config["experiment"]["budget_minutes_per_model_category"]

    # El presupuesto de cada modelo y categoría se reparte
    # equitativamente entre sus dos variantes.
    variant_budget_seconds = (
        budget_minutes * 60 / len(variants)
        if budget_minutes is not None
        else float("inf")
    )

    experiment_results = []

    for variant_name in variants:
        print(
            f"\nModelo: {model_name} | "
            f"Categoría: {category} | "
            f"Variante: {variant_name}",
            flush=True,
        )

        set_experiment_seed(config["project"]["seed"])

        # Crear los DataLoader para cada variante garantiza
        # que empiecen con la misma secuencia aleatoria.
        training_loader, validation_loader = create_data_loaders(
            category,
            model_name,
            config,
        )

        detector = create_detector(
            model_name,
            variant_name,
            config,
        )

        output_dir = (
            PROJECT_ROOT
            / config["results"]["runs"]
            / model_name
            / category
            / variant_name
        )

        output_dir.mkdir(parents=True, exist_ok=True)

        checkpoint_path = output_dir / "model.pt"
        results_path = output_dir / "metrics.json"

        synchronize_device(detector.device)

        fit_start = time.perf_counter()
        deadline = fit_start + variant_budget_seconds

        if model_name == "patchcore":
            fit_details = detector.fit(training_loader)

            detector.save(checkpoint_path)

            training_history = []
            best_epoch = None
            best_validation_auroc = None
            stop_reason = "completed"

        else:
            training_result = train_detector(
                detector=detector,
                model_name=model_name,
                training_loader=training_loader,
                validation_loader=validation_loader,
                config=config,
                deadline=deadline,
                checkpoint_path=checkpoint_path,
            )

            detector = training_result["detector"]
            training_history = training_result["history"]
            best_epoch = training_result["best_epoch"]
            best_validation_auroc = training_result["best_validation_image_auroc"]
            stop_reason = training_result["stop_reason"]

            fit_details = {}

        synchronize_device(detector.device)

        fit_seconds = time.perf_counter() - fit_start

        # La calibración final se mide por separado.
        validation_start = time.perf_counter()

        validation_metrics = evaluate_validation(
            detector,
            validation_loader,
            config,
        )

        validation_seconds = time.perf_counter() - validation_start

        results = {
            "category": category,
            "model": model_name,
            "variant": variant_name,
            "seed": config["project"]["seed"],
            "image_size": config["data"]["image_size"],
            "model_configuration": settings,
            "variant_configuration": variants[variant_name],
            "best_epoch": best_epoch,
            "best_validation_image_auroc": best_validation_auroc,
            "stop_reason": stop_reason,
            "fit_seconds": round(fit_seconds, 3),
            "validation_seconds": round(validation_seconds, 3),
            "total_seconds": round(fit_seconds + validation_seconds, 3),
            "budget_minutes": (
                variant_budget_seconds / 60
                if np.isfinite(variant_budget_seconds)
                else None
            ),
            "budget_exceeded": (fit_seconds > variant_budget_seconds),
            "fit_details": fit_details,
            "training_history": training_history,
            "validation": validation_metrics,
        }

        results_path.write_text(
            json.dumps(
                results,
                indent=2,
                ensure_ascii=False,
                allow_nan=False,
            ),
            encoding="utf-8",
        )

        experiment_results.append(results)

        print(
            "Image-AUROC:",
            f"{validation_metrics['detection']['image_auroc']:.4f}",
        )
        print(
            "AUPRO:",
            f"{validation_metrics['segmentation']['aupro']:.4f}",
        )
        print(f"Mejor época: {best_epoch}")
        print(f"Tiempo de ajuste: {fit_seconds:.2f} s")
        print(f"Tiempo de validación: {validation_seconds:.2f} s")
        print(f"Resultados: {results_path}")

    return experiment_results


def main():
    parser = argparse.ArgumentParser(description="Ejecuta experimentos de MVTec AD.")

    parser.add_argument(
        "--category",
        required=True,
        help="Categoría, por ejemplo bottle.",
    )

    parser.add_argument(
        "--model",
        required=True,
        choices=["autoencoder", "patchcore", "dual_unet"],
        help="Detector a ejecutar.",
    )

    args = parser.parse_args()
    config = load_experiment_config()

    run_experiment(
        category=args.category,
        model_name=args.model,
        config=config,
    )


if __name__ == "__main__":
    main()
