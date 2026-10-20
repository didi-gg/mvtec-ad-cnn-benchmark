"""Detección de anomalías con PatchCore y ResNet18."""

from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torchvision.models import ResNet18_Weights, resnet18
from tqdm.auto import tqdm

from src.models.base import AnomalyDetector


class PatchCoreDetector(AnomalyDetector):
    """Detector basado en características normales y vecinos cercanos."""

    def __init__(
        self,
        memory_fraction=0.01,
        device="mps",
        seed=1926,
        pretrained=True,
    ):
        if not 0 < memory_fraction <= 1:
            raise ValueError("memory_fraction debe estar entre 0 y 1.")

        self.memory_fraction = memory_fraction
        self.device = torch.device(device)
        self.seed = seed

        weights = ResNet18_Weights.DEFAULT if pretrained else None
        self.backbone = resnet18(weights=weights).to(self.device)
        self.backbone.eval()
        self.backbone.requires_grad_(False)

        self.memory_bank = None

        # Normalización utilizada por ResNet18 preentrenada.
        self.image_mean = torch.tensor(
            [0.485, 0.456, 0.406],
            device=self.device,
        ).view(1, 3, 1, 1)

        self.image_std = torch.tensor(
            [0.229, 0.224, 0.225],
            device=self.device,
        ).view(1, 3, 1, 1)

    def extract_patch_features(self, images):
        """Extrae características de layer2 y layer3 de ResNet18."""
        images = images.to(self.device)
        normalized_images = (images - self.image_mean) / self.image_std

        with torch.inference_mode():
            features = self.backbone.conv1(normalized_images)
            features = self.backbone.bn1(features)
            features = self.backbone.relu(features)
            features = self.backbone.maxpool(features)

            features = self.backbone.layer1(features)
            layer2_features = self.backbone.layer2(features)
            layer3_features = self.backbone.layer3(layer2_features)

            # Misma resolución espacial para las dos capas.
            layer3_features = F.interpolate(
                layer3_features,
                size=layer2_features.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )

            combined_features = torch.cat(
                [layer2_features, layer3_features],
                dim=1,
            )

            # Suavizado local de las características.
            combined_features = F.avg_pool2d(
                combined_features,
                kernel_size=3,
                stride=1,
                padding=1,
            )

            return combined_features

    def select_memory_bank_features(self, features):
        """Selecciona un coreset aproximado con distancias calculadas en MPS."""
        total_features = len(features)
        selected_count = max(1, round(total_features * self.memory_fraction))

        if selected_count == total_features:
            return features

        rng = np.random.default_rng(self.seed)

        # Se limita el conjunto de candidatos para controlar el costo.
        candidate_count = min(
            total_features,
            max(10000, selected_count * 2),
        )

        candidate_indices = rng.choice(
            total_features,
            size=candidate_count,
            replace=False,
        )

        # La selección se realiza sobre una proyección de 16 dimensiones.
        candidates = features[candidate_indices].to(self.device)

        projection = torch.from_numpy(
            rng.normal(size=(candidates.shape[1], 16)).astype(np.float32)
        ).to(self.device)

        projected = candidates @ projection / np.sqrt(16)

        del candidates

        minimum_distances = torch.full(
            (candidate_count,),
            float("inf"),
            device=self.device,
        )

        selected_positions = torch.empty(
            selected_count,
            dtype=torch.long,
            device=self.device,
        )

        selected_flags = torch.zeros(
            candidate_count,
            dtype=torch.bool,
            device=self.device,
        )

        current_position = int(rng.integers(candidate_count))

        with torch.no_grad():
            for index in tqdm(
                range(selected_count),
                desc=f"Coreset {self.memory_fraction:.0%}",
            ):
                selected_positions[index] = current_position
                selected_flags[current_position] = True

                difference = projected - projected[current_position]

                distances = (difference * difference).sum(dim=1)

                minimum_distances = torch.minimum(
                    minimum_distances,
                    distances,
                )

                minimum_distances[selected_flags] = -1

                # Se sincroniza para obtener el índice en Python.
                current_position = int(minimum_distances.argmax().item())

        original_indices = candidate_indices[selected_positions.cpu().numpy()]

        return features[original_indices].contiguous()

    def fit(self, train_loader):
        """Construye el banco usando únicamente imágenes normales."""
        self.backbone.eval()
        extracted_features = []

        for batch in train_loader:
            if "label" in batch and torch.any(batch["label"] != 0):
                raise ValueError("PatchCore debe ajustarse con imágenes normales.")

            feature_maps = self.extract_patch_features(batch["image"])

            # [B, C, H, W] -> [B*H*W, C]
            patch_vectors = feature_maps.permute(0, 2, 3, 1).reshape(
                -1, feature_maps.shape[1]
            )

            extracted_features.append(patch_vectors.cpu().contiguous())

        if not extracted_features:
            raise ValueError("El conjunto de entrenamiento está vacío.")

        all_features = torch.cat(extracted_features, dim=0)

        print(f"Características extraídas: {len(all_features):,}")

        selected_features = self.select_memory_bank_features(all_features)

        # El banco se almacena en CPU y se utiliza en MPS
        # durante la predicción.
        self.memory_bank = selected_features.cpu()

        print(
            f"Banco de memoria: {len(self.memory_bank):,} "
            f"características ({self.memory_fraction:.0%})"
        )

        return {
            "total_features": len(all_features),
            "retained_features": len(self.memory_bank),
        }

    def find_nearest_normal_distances(self, query_features):
        """Calcula la distancia mínima al banco para cada parche."""
        if self.memory_bank is None:
            raise RuntimeError("Debe ejecutar fit() o load() primero.")

        memory_bank = self.memory_bank.to(self.device)
        nearest_distances = []

        # Bloques para evitar matrices de distancias demasiado grandes.
        query_chunk_size = 256
        memory_chunk_size = 4096

        with torch.inference_mode():
            for query_chunk in query_features.split(query_chunk_size):
                minimum_distances = torch.full(
                    (len(query_chunk),),
                    float("inf"),
                    device=self.device,
                )

                for memory_chunk in memory_bank.split(memory_chunk_size):
                    distances = torch.cdist(
                        query_chunk.unsqueeze(0),
                        memory_chunk.unsqueeze(0),
                    ).squeeze(0)

                    minimum_distances = torch.minimum(
                        minimum_distances,
                        distances.min(dim=1).values,
                    )

                nearest_distances.append(minimum_distances)

        return torch.cat(nearest_distances)

    def predict_anomaly_map(self, images):
        """Devuelve un mapa de anomalías [B, 1, H, W]."""
        if self.memory_bank is None:
            raise RuntimeError("El detector aún no está ajustado.")

        images = images.to(self.device)

        feature_maps = self.extract_patch_features(images)

        batch_size, channels, height, width = feature_maps.shape

        query_features = feature_maps.permute(0, 2, 3, 1).reshape(-1, channels)

        distances = self.find_nearest_normal_distances(query_features)

        anomaly_maps = distances.reshape(batch_size, 1, height, width)

        return F.interpolate(
            anomaly_maps,
            size=images.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )

    def save(self, path):
        """Guarda el backbone, la configuración y el banco de memoria."""
        if self.memory_bank is None:
            raise RuntimeError("No se puede guardar un detector sin ajustar.")

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        torch.save(
            {
                "backbone_state_dict": self.backbone.state_dict(),
                "memory_bank": self.memory_bank,
                "memory_fraction": self.memory_fraction,
                "seed": self.seed,
            },
            path,
        )

    @classmethod
    def load(cls, path, device="cpu"):
        """Reconstruye un detector desde un checkpoint."""
        checkpoint = torch.load(
            path,
            map_location="cpu",
            weights_only=True,
        )

        detector = cls(
            memory_fraction=checkpoint["memory_fraction"],
            seed=checkpoint["seed"],
            device=device,
            pretrained=False,
        )

        detector.backbone.load_state_dict(checkpoint["backbone_state_dict"])
        detector.memory_bank = checkpoint["memory_bank"]

        return detector
