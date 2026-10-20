from abc import ABC, abstractmethod


class AnomalyDetector(ABC):
    """Interfaz común para los detectores de anomalías."""

    @abstractmethod
    def fit(self, train_loader):
        """Ajusta el detector utilizando imágenes de entrenamiento."""

    @abstractmethod
    def predict_anomaly_map(self, images):
        """Devuelve un mapa de anomalías por imagen [B, 1, H, W]."""

    def predict_image_scores(self, images):
        """Calcula una puntuación de anomalía por imagen."""
        anomaly_maps = self.predict_anomaly_map(images)
        return anomaly_maps.flatten(start_dim=1).amax(dim=1)

    @abstractmethod
    def save(self, path):
        """Guarda el estado necesario para reutilizar el detector."""

    @classmethod
    @abstractmethod
    def load(cls, path, device):
        """Reconstruye un detector previamente guardado."""
