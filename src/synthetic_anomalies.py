"""Generación reproducible de anomalías sintéticas."""

import numpy as np
import torch
import torch.nn.functional as F


def generate_cutpaste_anomaly(image, rng):
    """Introduce una anomalía rectangular mediante recorte y pegado."""
    _, height, width = image.shape

    patch_height = int(rng.integers(height // 10, height // 4))
    patch_width = int(rng.integers(width // 10, width // 4))

    source_y = int(rng.integers(0, height - patch_height + 1))
    source_x = int(rng.integers(0, width - patch_width + 1))

    target_y = int(rng.integers(0, height - patch_height + 1))
    target_x = int(rng.integers(0, width - patch_width + 1))

    patch = image[
        :,
        source_y : source_y + patch_height,
        source_x : source_x + patch_width,
    ].clone()

    modified_image = image.clone()
    modified_image[
        :,
        target_y : target_y + patch_height,
        target_x : target_x + patch_width,
    ] = patch

    mask = torch.zeros((1, height, width), dtype=torch.float32)
    mask[
        :,
        target_y : target_y + patch_height,
        target_x : target_x + patch_width,
    ] = 1

    return modified_image, mask


def generate_irregular_texture_anomaly(image, rng):
    """Introduce una textura distinta dentro de una región irregular."""
    channels, height, width = image.shape

    # Crear una máscara irregular a partir de ruido suavizado.
    coarse_height = max(2, height // 32)
    coarse_width = max(2, width // 32)

    noise = torch.from_numpy(
        rng.random((1, 1, coarse_height, coarse_width)).astype(np.float32)
    )

    smooth_noise = F.interpolate(
        noise,
        size=(height, width),
        mode="bicubic",
        align_corners=False,
    )[0]

    mask = (smooth_noise > 0.65).float()

    # Crear una textura aleatoria con variaciones espaciales.
    texture = torch.from_numpy(rng.random((channels, height, width)).astype(np.float32))

    texture = F.avg_pool2d(
        texture.unsqueeze(0),
        kernel_size=7,
        stride=1,
        padding=3,
    )[0]

    # Mezclar la textura con la imagen dentro de la máscara.
    modified_image = image * (1 - mask) + (0.3 * image + 0.7 * texture) * mask

    return modified_image.clamp(0, 1), mask


def generate_synthetic_anomaly(image, method, seed):
    """Genera una anomalía sintética reproducible."""
    rng = np.random.default_rng(seed)

    generators = {
        "cutpaste": generate_cutpaste_anomaly,
        "irregular_texture": generate_irregular_texture_anomaly,
    }

    if method not in generators:
        raise ValueError(f"Método de anomalía desconocido: {method}")

    return generators[method](image, rng)
