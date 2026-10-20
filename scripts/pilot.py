"""
Piloto de rendimiento con imágenes reales de MVTec AD.

Ejecuta:
    python scripts/pilot.py --category bottle
    python scripts/pilot.py --category carpet
    python scripts/pilot.py --category toothbrush

Mide:
- Autoencoder CNN: entrenamiento e inferencia.
- PatchCore: ResNet18, extracción de características y búsqueda.
- Dos U-Net: entrenamiento e inferencia de dos capacidades.

No utiliza imágenes ni máscaras del test oficial.
"""

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import psutil
import torch
import torch.nn.functional as F
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision.models import (
    ResNet18_Weights,
    resnet18,
)
from torchvision.models.feature_extraction import create_feature_extractor
from torchvision.transforms import v2

ROOT = Path(__file__).resolve().parents[1]
SPLITS = ROOT / "results" / "splits"
OUTPUT = ROOT / "results" / "hardware"

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

IMAGE_SIZE = 256
SEED = 1926


def sync():
    if DEVICE.type == "mps":
        torch.mps.synchronize()


def memory_snapshot():
    result = {"process_rss_gb": round(psutil.Process().memory_info().rss / 2**30, 3)}
    if DEVICE.type == "mps":
        result["mps_driver_gb"] = round(torch.mps.driver_allocated_memory() / 2**30, 3)
    return result


class NormalImages(Dataset):
    def __init__(self, root, paths):
        self.paths = [root / p for p in paths]
        self.transform = v2.Compose(
            [
                v2.Resize((IMAGE_SIZE, IMAGE_SIZE)),
                v2.ToImage(),
                v2.ToDtype(torch.float32, scale=True),
            ]
        )

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        with Image.open(self.paths[index]) as image:
            return self.transform(image.convert("RGB"))


def double_conv(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, 3, padding=1),
        nn.ReLU(inplace=True),
    )


class UNet(nn.Module):
    def __init__(self, input_channels, output_channels, channels):
        super().__init__()

        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.pools = nn.ModuleList()

        previous = input_channels

        for width in channels:
            self.encoders.append(double_conv(previous, width))
            previous = width

        for width in reversed(channels[:-1]):
            self.decoders.append(double_conv(previous + width, width))
            previous = width

        self.output = nn.Conv2d(previous, output_channels, 1)

    def forward(self, x):
        skips = []

        for i, encoder in enumerate(self.encoders):
            x = encoder(x)
            if i < len(self.encoders) - 1:
                skips.append(x)
                x = F.avg_pool2d(x, 2)

        for decoder, skip in zip(self.decoders, reversed(skips)):
            x = F.interpolate(x, size=skip.shape[-2:], mode="nearest")
            x = decoder(torch.cat([x, skip], dim=1))

        return self.output(x)


class Autoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        # Encoder-decoder convolucional sin conexiones de salto.
        channels = [16, 32, 64]
        layers = []
        previous = 3

        for width in channels:
            layers.extend(
                [
                    nn.Conv2d(previous, width, 3, padding=1),
                    nn.ReLU(),
                    nn.AvgPool2d(2),
                ]
            )
            previous = width

        for width in [32, 16, 3]:
            layers.extend(
                [
                    nn.Upsample(scale_factor=2, mode="nearest"),
                    nn.Conv2d(previous, width, 3, padding=1),
                ]
            )
            if width != 3:
                layers.append(nn.ReLU())
            previous = width

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class DualUNet(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.reconstructor = UNet(3, 3, channels)
        self.segmenter = UNet(6, 1, channels)

    def forward(self, image):
        reconstruction = self.reconstructor(image)
        logits = self.segmenter(torch.cat([image, reconstruction], dim=1))
        return reconstruction, logits


def synthetic_defects(images):
    """
    Generación simple para medir entrenamiento.
    No es todavía el generador experimental definitivo.
    """
    modified = images.clone()
    masks = torch.zeros_like(images[:, :1])

    height, width = images.shape[-2:]
    patch = max(8, height // 5)

    for i in range(images.shape[0]):
        y = int(torch.randint(0, height - patch, (1,)))
        x = int(torch.randint(0, width - patch, (1,)))

        source = torch.roll(
            images[i],
            shifts=(height // 3, width // 3),
            dims=(-2, -1),
        )

        modified[i, :, y : y + patch, x : x + patch] = source[
            :, y : y + patch, x : x + patch
        ]
        masks[i, :, y : y + patch, x : x + patch] = 1

    return modified, masks


def benchmark_train(name, model, loader, dual=False):
    model = model.to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    def epoch():
        model.train()
        losses = []

        for images in loader:
            images = images.to(DEVICE)
            optimizer.zero_grad(set_to_none=True)

            if dual:
                altered, masks = synthetic_defects(images)
                reconstruction, logits = model(altered)

                loss = F.l1_loss(
                    reconstruction, images
                ) + F.binary_cross_entropy_with_logits(logits, masks)
            else:
                reconstruction = model(images)
                loss = F.mse_loss(reconstruction, images)

            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        sync()
        return float(np.mean(losses))

    start = time.perf_counter()
    loss = epoch()
    epoch_seconds = time.perf_counter() - start
    memory = memory_snapshot()

    model.eval()
    image = next(iter(loader))[:1].to(DEVICE)

    with torch.inference_mode():
        for _ in range(5):
            model(image)
        sync()

        times = []
        for _ in range(20):
            start = time.perf_counter()
            model(image)
            sync()
            times.append(time.perf_counter() - start)

    result = {
        "workload": name,
        "parameters": sum(p.numel() for p in model.parameters()),
        "train_images": len(loader.dataset),
        "batch_size": loader.batch_size,
        "epoch_seconds": round(epoch_seconds, 3),
        "training_loss": round(loss, 6),
        "inference_median_ms": round(np.median(times) * 1000, 3),
        "inference_p95_ms": round(np.percentile(times, 95) * 1000, 3),
        "memory_snapshot": memory,
    }

    del model, optimizer, image
    if DEVICE.type == "mps":
        torch.mps.empty_cache()

    return result


def patch_embeddings(extractor, images):
    features = extractor(images)

    layer2 = features["layer2"]
    layer3 = F.interpolate(
        features["layer3"],
        size=layer2.shape[-2:],
        mode="nearest",
    )

    embeddings = torch.cat([layer2, layer3], dim=1)

    return embeddings.permute(0, 2, 3, 1).reshape(-1, embeddings.shape[1])


def benchmark_patchcore(loader, validation_loader):
    # Los pesos se descargan una vez si no están en caché.
    backbone = resnet18(weights=ResNet18_Weights.DEFAULT)

    extractor = (
        create_feature_extractor(
            backbone,
            return_nodes={
                "layer2": "layer2",
                "layer3": "layer3",
            },
        )
        .to(DEVICE)
        .eval()
    )

    # Normalización ImageNet requerida por los pesos.
    mean = torch.tensor([0.485, 0.456, 0.406], device=DEVICE).view(1, 3, 1, 1)

    std = torch.tensor([0.229, 0.224, 0.225], device=DEVICE).view(1, 3, 1, 1)

    banks = []
    start = time.perf_counter()

    with torch.inference_mode():
        for images in loader:
            images = (images.to(DEVICE) - mean) / std
            embeddings = patch_embeddings(extractor, images)
            banks.append(embeddings.cpu())

    sync()
    extraction_seconds = time.perf_counter() - start

    all_features = torch.cat(banks)
    total = len(all_features)

    # Se mide muestreo aleatorio para dimensionar búsqueda y memoria.
    # El coreset aproximado definitivo se perfilará por separado.
    rng = np.random.default_rng(SEED)
    permutation = torch.from_numpy(rng.permutation(total))

    query_images = next(iter(validation_loader))[:1]
    query_images = (query_images.to(DEVICE) - mean) / std

    results = []

    with torch.inference_mode():
        for fraction in [0.01, 0.05]:
            selected = max(1, round(total * fraction))

            start = time.perf_counter()
            bank = all_features[permutation[:selected]].to(DEVICE)
            sync()
            bank_seconds = time.perf_counter() - start

            # Calentamiento
            query_features = patch_embeddings(extractor, query_images)
            torch.cdist(query_features, bank).min(dim=1)
            sync()

            times = []
            for _ in range(5):
                start = time.perf_counter()

                query_features = patch_embeddings(extractor, query_images)
                distances = torch.cdist(query_features, bank)
                distances.min(dim=1)

                sync()
                times.append(time.perf_counter() - start)

            results.append(
                {
                    "fraction": fraction,
                    "total_features": total,
                    "retained_features": selected,
                    "feature_extraction_seconds": round(extraction_seconds, 3),
                    "bank_sampling_seconds": round(bank_seconds, 3),
                    "bank_size_mb": round(
                        bank.numel() * bank.element_size() / 2**20, 2
                    ),
                    "inference_median_ms": round(np.median(times) * 1000, 3),
                    "inference_p95_ms": round(np.percentile(times, 95) * 1000, 3),
                    "memory_snapshot": memory_snapshot(),
                }
            )

            del bank
            if DEVICE.type == "mps":
                torch.mps.empty_cache()

    del extractor, backbone, all_features, banks

    return {
        "workload": "patchcore_resnet18",
        "sampling_method": "random_pilot_only",
        "variants": results,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--category", default="bottle")
    args = parser.parse_args()

    torch.manual_seed(SEED)
    np.random.seed(SEED)

    manifest = json.loads((SPLITS / f"{args.category}.json").read_text())

    data_root = ROOT / "data" / "mvtec_anomaly_detection"

    train_data = NormalImages(data_root, manifest["train"])
    val_data = NormalImages(data_root, manifest["validation"])

    def loader(dataset, batch):
        return DataLoader(
            dataset,
            batch_size=batch,
            shuffle=False,
            num_workers=0,
        )

    print(f"Categoría: {args.category}")
    print(f"Dispositivo: {DEVICE}")
    print(f"Train: {len(train_data)}")
    print(f"Validación: {len(val_data)}")

    results = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "category": args.category,
        "device": str(DEVICE),
        "image_size": IMAGE_SIZE,
        "measurements": [],
    }

    print("\n1. Autoencoder CNN")
    results["measurements"].append(
        benchmark_train(
            "autoencoder",
            Autoencoder(),
            loader(train_data, 8),
        )
    )

    print("2. PatchCore + ResNet18")
    results["measurements"].append(
        benchmark_patchcore(
            loader(train_data, 4),
            loader(val_data, 1),
        )
    )

    print("3. Dos U-Net pequeñas")
    results["measurements"].append(
        benchmark_train(
            "dual_unet_small",
            DualUNet([16, 32, 64]),
            loader(train_data, 2),
            dual=True,
        )
    )

    print("4. Dos U-Net grandes")
    results["measurements"].append(
        benchmark_train(
            "dual_unet_large",
            DualUNet([16, 32, 64, 128]),
            loader(train_data, 2),
            dual=True,
        )
    )

    OUTPUT.mkdir(parents=True, exist_ok=True)
    output = OUTPUT / f"pilot_{args.category}.json"

    output.write_text(
        json.dumps(results, indent=2),
        encoding="utf-8",
    )

    print("\nResumen:")
    for item in results["measurements"]:
        if "epoch_seconds" in item:
            print(f"{item['workload']}: {item['epoch_seconds']:.2f} s/época")
        else:
            print(
                "PatchCore extracción:",
                round(
                    item["variants"][0]["feature_extraction_seconds"],
                    2,
                ),
                "s",
            )

    print("\nResultados:", output)


if __name__ == "__main__":
    main()
