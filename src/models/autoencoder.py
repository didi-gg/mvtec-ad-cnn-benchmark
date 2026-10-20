"""Autoencoder convolucional para detección de anomalías industriales."""

import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

from src.models.base import AnomalyDetector


class ConvolutionalAutoencoder(nn.Module):
    """Red encoder-decoder que reconstruye imágenes RGB."""

    def __init__(self, channels=(16, 32, 64)):
        super().__init__()

        self.channels = tuple(channels)

        encoder_layers = []
        input_channels = 3

        for output_channels in self.channels:
            encoder_layers.extend(
                [
                    nn.Conv2d(input_channels, output_channels, 3, padding=1),
                    nn.ReLU(),
                    nn.AvgPool2d(2),
                ]
            )
            input_channels = output_channels

        decoder_layers = []
        decoder_channels = list(reversed(self.channels[:-1])) + [3]

        for output_channels in decoder_channels:
            decoder_layers.extend(
                [
                    nn.Upsample(scale_factor=2, mode="nearest"),
                    nn.Conv2d(input_channels, output_channels, 3, padding=1),
                ]
            )

            if output_channels != 3:
                decoder_layers.append(nn.ReLU())

            input_channels = output_channels

        self.encoder = nn.Sequential(*encoder_layers)
        self.decoder = nn.Sequential(*decoder_layers)

    def forward(self, images):
        """Reconstruye un batch de imágenes."""
        features = self.encoder(images)
        return self.decoder(features)


class AutoencoderDetector(AnomalyDetector):
    """Detector de anomalías basado en error de reconstrucción."""

    def __init__(
        self,
        loss_name="mse",
        channels=(16, 32, 64),
        learning_rate=0.001,
        epochs=100,
        device="mps",
    ):
        if loss_name not in {"mse", "l1"}:
            raise ValueError("loss_name debe ser 'mse' o 'l1'.")

        self.loss_name = loss_name
        self.channels = tuple(channels)
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.device = torch.device(device)

        self.network = ConvolutionalAutoencoder(channels=self.channels).to(self.device)

        self.training_history = []

    def compute_reconstruction_loss(self, images, reconstructions):
        """Calcula MSE o L1 según la variante experimental."""
        if self.loss_name == "mse":
            return F.mse_loss(reconstructions, images)
        return F.l1_loss(reconstructions, images)

    def fit(self, train_loader):
        """Entrena el autoencoder únicamente con imágenes normales."""
        optimizer = torch.optim.Adam(
            self.network.parameters(),
            lr=self.learning_rate,
        )

        self.training_history = []

        for epoch in range(self.epochs):
            self.network.train()

            total_loss = 0.0
            total_images = 0
            start_time = time.perf_counter()

            for batch in train_loader:
                images = batch["image"].to(self.device)

                optimizer.zero_grad(set_to_none=True)

                reconstructions = self.network(images)
                loss = self.compute_reconstruction_loss(images, reconstructions)

                loss.backward()
                optimizer.step()

                batch_size = images.shape[0]
                total_loss += loss.item() * batch_size
                total_images += batch_size

            if self.device.type == "mps":
                torch.mps.synchronize()

            epoch_loss = total_loss / total_images
            epoch_seconds = time.perf_counter() - start_time

            self.training_history.append(
                {
                    "epoch": epoch + 1,
                    "loss": epoch_loss,
                    "seconds": epoch_seconds,
                }
            )

            print(
                f"Época {epoch + 1}/{self.epochs} | "
                f"Pérdida: {epoch_loss:.6f} | "
                f"Tiempo: {epoch_seconds:.2f}s"
            )

        return self.training_history

    def reconstruct_images(self, images):
        """Reconstruye imágenes sin calcular gradientes."""
        self.network.eval()

        with torch.inference_mode():
            images = images.to(self.device)
            return self.network(images)

    def predict_anomaly_map(self, images):
        """Calcula un mapa de error absoluto de tamaño [B, 1, H, W]."""
        images = images.to(self.device)
        reconstructions = self.reconstruct_images(images)

        anomaly_maps = torch.abs(images - reconstructions).mean(dim=1, keepdim=True)

        return anomaly_maps

    def save(self, path):
        """Guarda pesos y configuración para reproducir el detector."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        torch.save(
            {
                "network_state_dict": self.network.state_dict(),
                "loss_name": self.loss_name,
                "channels": self.channels,
                "learning_rate": self.learning_rate,
                "epochs": self.epochs,
                "training_history": self.training_history,
            },
            path,
        )

    @classmethod
    def load(cls, path, device="cpu"):
        """Reconstruye un detector desde un checkpoint."""
        checkpoint = torch.load(
            path,
            map_location=device,
            weights_only=True,
        )

        detector = cls(
            loss_name=checkpoint["loss_name"],
            channels=checkpoint["channels"],
            learning_rate=checkpoint["learning_rate"],
            epochs=checkpoint["epochs"],
            device=device,
        )

        detector.network.load_state_dict(checkpoint["network_state_dict"])
        detector.training_history = checkpoint["training_history"]

        return detector
