"""Detector de anomalías mediante reconstrucción y segmentación con dos U-Net."""

import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from src.models.base import AnomalyDetector
from src.synthetic_anomalies import generate_synthetic_anomaly


class UNet(nn.Module):
    """Red U-Net configurable para reconstrucción o segmentación."""

    def __init__(self, input_channels, output_channels, channels):
        super().__init__()

        def convolution_block(in_channels, out_channels):
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 3, padding=1),
                nn.ReLU(),
                nn.Conv2d(out_channels, out_channels, 3, padding=1),
                nn.ReLU(),
            )

        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()

        previous_channels = input_channels

        for width in channels:
            self.encoders.append(convolution_block(previous_channels, width))
            previous_channels = width

        for width in reversed(channels[:-1]):
            self.decoders.append(convolution_block(previous_channels + width, width))
            previous_channels = width

        self.output_layer = nn.Conv2d(previous_channels, output_channels, 1)

    def forward(self, images):
        features = images
        skip_connections = []

        for index, encoder in enumerate(self.encoders):
            features = encoder(features)

            if index < len(self.encoders) - 1:
                skip_connections.append(features)
                features = F.avg_pool2d(features, 2)

        for decoder, skip_features in zip(self.decoders, reversed(skip_connections)):
            features = F.interpolate(
                features,
                size=skip_features.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
            features = decoder(torch.cat([features, skip_features], dim=1))

        return self.output_layer(features)


class DualUNetDetector(AnomalyDetector):
    """Entrena dos U-Net para reconstruir imágenes y localizar anomalías."""

    def __init__(
        self,
        channels=(16, 32, 64),
        learning_rate=0.0001,
        epochs=100,
        device="mps",
        seed=1926,
    ):
        self.channels = tuple(channels)
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.device = torch.device(device)
        self.seed = seed

        self.reconstruction_network = UNet(
            input_channels=3,
            output_channels=3,
            channels=self.channels,
        ).to(self.device)

        self.segmentation_network = UNet(
            input_channels=6,
            output_channels=1,
            channels=self.channels,
        ).to(self.device)

        self.training_history = []

    def reconstruct_images(self, images):
        """Reconstruye una imagen normal a partir de una imagen alterada."""
        return torch.sigmoid(self.reconstruction_network(images))

    def predict_segmentation_logits(self, images):
        """Predice los logits de las regiones anómalas."""
        reconstruction = self.reconstruct_images(images)

        return self.segmentation_network(torch.cat([images, reconstruction], dim=1))

    def generate_training_anomalies(self, images, rng):
        """Genera defectos exclusivamente sobre imágenes de entrenamiento."""
        modified_images = []
        masks = []
        methods = ("cutpaste", "irregular_texture")

        for image in images:
            method = methods[int(rng.integers(len(methods)))]
            seed = int(rng.integers(0, 2**32))

            modified_image, mask = generate_synthetic_anomaly(
                image, method=method, seed=seed
            )

            modified_images.append(modified_image)
            masks.append(mask)

        return torch.stack(modified_images), torch.stack(masks)

    def compute_segmentation_loss(self, logits, masks):
        """Calcula BCE + Dice para el entrenamiento de segmentación."""
        bce = F.binary_cross_entropy_with_logits(logits, masks)
        probabilities = torch.sigmoid(logits)

        intersection = (probabilities * masks).sum(dim=(1, 2, 3))
        denominator = (probabilities + masks).sum(dim=(1, 2, 3))

        dice = 1 - ((2 * intersection + 1e-6) / (denominator + 1e-6)).mean()

        return bce + dice

    def fit(self, train_loader):
        """Entrena ambas U-Net utilizando anomalías sintéticas."""
        optimizer = torch.optim.Adam(
            list(self.reconstruction_network.parameters())
            + list(self.segmentation_network.parameters()),
            lr=self.learning_rate,
        )

        rng = np.random.default_rng(self.seed)
        self.training_history = []

        for epoch in range(self.epochs):
            self.reconstruction_network.train()
            self.segmentation_network.train()

            total_loss = 0.0
            total_images = 0

            if self.device.type == "mps":
                torch.mps.synchronize()

            start_time = time.perf_counter()

            for batch in train_loader:
                normal_images = batch["image"]

                modified_images, masks = self.generate_training_anomalies(
                    normal_images, rng
                )

                normal_images = normal_images.to(self.device)
                modified_images = modified_images.to(self.device)
                masks = masks.to(self.device)

                optimizer.zero_grad(set_to_none=True)

                reconstruction = self.reconstruct_images(modified_images)
                logits = self.segmentation_network(
                    torch.cat([modified_images, reconstruction], dim=1)
                )

                reconstruction_loss = F.l1_loss(reconstruction, normal_images)
                segmentation_loss = self.compute_segmentation_loss(logits, masks)

                loss = reconstruction_loss + segmentation_loss
                loss.backward()
                optimizer.step()

                batch_size = normal_images.shape[0]
                total_loss += loss.item() * batch_size
                total_images += batch_size

            if self.device.type == "mps":
                torch.mps.synchronize()

            epoch_seconds = time.perf_counter() - start_time

            self.training_history.append(
                {
                    "epoch": epoch + 1,
                    "loss": total_loss / total_images,
                    "seconds": epoch_seconds,
                }
            )

            print(
                f"Época {epoch + 1}/{self.epochs} | "
                f"Pérdida: {total_loss / total_images:.6f} | "
                f"Tiempo: {epoch_seconds:.2f}s"
            )

        return self.training_history

    def predict_anomaly_map(self, images):
        """Devuelve mapas de anomalías [B, 1, H, W]."""
        self.reconstruction_network.eval()
        self.segmentation_network.eval()

        with torch.inference_mode():
            images = images.to(self.device)
            logits = self.predict_segmentation_logits(images)
            return torch.sigmoid(logits)

    def save(self, path):
        """Guarda ambas redes y sus parámetros de configuración."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        torch.save(
            {
                "reconstruction_state_dict": self.reconstruction_network.state_dict(),
                "segmentation_state_dict": self.segmentation_network.state_dict(),
                "channels": self.channels,
                "learning_rate": self.learning_rate,
                "epochs": self.epochs,
                "seed": self.seed,
                "training_history": self.training_history,
            },
            path,
        )

    @classmethod
    def load(cls, path, device="cpu"):
        """Carga el detector completo desde un checkpoint."""
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)

        detector = cls(
            channels=checkpoint["channels"],
            learning_rate=checkpoint["learning_rate"],
            epochs=checkpoint["epochs"],
            seed=checkpoint["seed"],
            device=device,
        )

        detector.reconstruction_network.load_state_dict(
            checkpoint["reconstruction_state_dict"]
        )
        detector.segmentation_network.load_state_dict(
            checkpoint["segmentation_state_dict"]
        )
        detector.training_history = checkpoint["training_history"]

        return detector
