"""Métricas comunes para detección y localización de anomalías."""

import numpy as np
import torch
from scipy.ndimage import label as label_regions
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)


def to_numpy(values):
    """Convierte tensores PyTorch o secuencias a NumPy."""
    if isinstance(values, torch.Tensor):
        return values.detach().cpu().numpy()
    return np.asarray(values)


def prepare_segmentation_arrays(anomaly_maps, masks):
    """Valida mapas y máscaras con dimensiones [N, H, W]."""
    scores = to_numpy(anomaly_maps).astype(np.float32)
    targets = to_numpy(masks)

    if scores.ndim == 4 and scores.shape[1] == 1:
        scores = scores[:, 0]

    if targets.ndim == 4 and targets.shape[1] == 1:
        targets = targets[:, 0]

    if scores.ndim != 3 or scores.shape != targets.shape:
        raise ValueError(
            "Los mapas y máscaras deben tener forma [N, H, W] "
            "o [N, 1, H, W] y las mismas dimensiones."
        )

    if not np.isfinite(scores).all():
        raise ValueError("Los mapas contienen valores no finitos.")

    return scores, targets > 0


def find_best_image_threshold(labels, scores):
    """Selecciona en validación el umbral que maximiza F1."""
    labels = to_numpy(labels).astype(int).reshape(-1)
    scores = to_numpy(scores).astype(float).reshape(-1)

    if labels.size == 0 or labels.size != scores.size:
        raise ValueError("Etiquetas y puntuaciones incompatibles.")

    if not np.isin(labels, [0, 1]).all():
        raise ValueError("Las etiquetas deben ser 0 o 1.")

    if not np.isfinite(scores).all():
        raise ValueError("Las puntuaciones deben ser finitas.")

    if not np.any(labels == 1):
        raise ValueError("Se necesitan anomalías de validación para seleccionar F1.")

    precision, recall, thresholds = precision_recall_curve(labels, scores)

    f1_values = (
        2
        * precision[:-1]
        * recall[:-1]
        / np.maximum(precision[:-1] + recall[:-1], 1e-12)
    )

    best_index = int(np.argmax(f1_values))
    return float(thresholds[best_index])


def find_best_pixel_threshold(anomaly_maps, masks, candidates=101):
    """Selecciona en validación el umbral que maximiza Dice global."""
    scores, targets = prepare_segmentation_arrays(anomaly_maps, masks)

    if not targets.any():
        raise ValueError("Se necesitan máscaras con anomalías para calibrar Dice.")

    thresholds = np.unique(
        np.quantile(
            scores.ravel(),
            np.linspace(0, 1, candidates),
        )
    )

    best_threshold = float(thresholds[0])
    best_dice = -1.0

    for threshold in thresholds:
        predicted = scores >= threshold

        intersection = np.count_nonzero(predicted & targets)
        denominator = predicted.sum() + targets.sum()

        dice = 2 * intersection / denominator if denominator > 0 else 1.0

        if dice > best_dice:
            best_dice = dice
            best_threshold = float(threshold)

    return best_threshold


def calculate_detection_metrics(labels, scores, threshold):
    """Calcula las métricas de clasificación por imagen."""
    labels = to_numpy(labels).astype(int).reshape(-1)
    scores = to_numpy(scores).astype(float).reshape(-1)

    if labels.size == 0 or labels.size != scores.size:
        raise ValueError("Etiquetas y puntuaciones incompatibles.")

    if not np.isin(labels, [0, 1]).all():
        raise ValueError("Las etiquetas deben ser 0 o 1.")

    if not np.isfinite(scores).all():
        raise ValueError("Las puntuaciones deben ser finitas.")

    predictions = (scores >= threshold).astype(int)

    image_auroc = (
        float(roc_auc_score(labels, scores))
        if np.unique(labels).size == 2
        else float("nan")
    )

    tn, fp, fn, tp = confusion_matrix(labels, predictions, labels=[0, 1]).ravel()

    return {
        "image_auroc": image_auroc,
        "precision": float(precision_score(labels, predictions, zero_division=0)),
        "recall": float(recall_score(labels, predictions, zero_division=0)),
        "f1": float(f1_score(labels, predictions, zero_division=0)),
        "confusion_matrix": {
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "tp": int(tp),
        },
    }


def calculate_aupro(anomaly_maps, masks, fpr_limit=0.30, points=100):
    """Calcula AUPRO normalizada hasta un FPR máximo.

    PRO: promedio del solapamiento de las regiones anómalas.
    FPR: proporción de píxeles normales clasificados como anomalía.
    """
    scores, targets = prepare_segmentation_arrays(anomaly_maps, masks)

    if not 0 < fpr_limit <= 1:
        raise ValueError("fpr_limit debe estar entre 0 y 1.")

    if not targets.any() or not (~targets).any():
        return float("nan")

    # Identificar todas las regiones anómalas del conjunto.
    region_ids = np.zeros(targets.shape, dtype=np.int32)
    total_regions = 0

    for image_index, mask in enumerate(targets):
        labeled, count = label_regions(mask, structure=np.ones((3, 3), dtype=int))

        labeled[labeled > 0] += total_regions
        region_ids[image_index] = labeled
        total_regions += count

    region_sizes = np.bincount(region_ids[targets], minlength=total_regions + 1)[1:]

    normal_scores = scores[~targets]
    anomaly_scores = scores[targets]

    # Muestrear umbrales de toda la distribución de puntuaciones.
    # Incluir explícitamente el punto con cero falsos positivos.
    thresholds = np.unique(
        np.concatenate(
            [
                np.quantile(scores, np.linspace(0, 1, points)),
                np.quantile(anomaly_scores, np.linspace(0, 1, points)),
                [np.nextafter(float(normal_scores.max()), np.inf)],
                [np.inf],
            ]
        )
    )

    fpr_values = []
    pro_values = []

    for threshold in thresholds:
        predictions = scores >= threshold

        false_positives = np.count_nonzero(predictions & ~targets)
        fpr = false_positives / normal_scores.size

        detected_per_region = np.bincount(
            region_ids[predictions],
            minlength=total_regions + 1,
        )[1:]

        pro = np.mean(detected_per_region / region_sizes)

        fpr_values.append(fpr)
        pro_values.append(pro)

    fpr_values = np.asarray(fpr_values)
    pro_values = np.asarray(pro_values)

    # Ordenar la curva por FPR y conservar el mayor PRO
    # cuando varios umbrales generan el mismo FPR.
    unique_fpr = np.unique(fpr_values)
    unique_pro = np.array(
        [pro_values[fpr_values == value].max() for value in unique_fpr]
    )

    # Interpolar el límite FPR=0.30 si no aparece exactamente.
    pro_at_limit = np.interp(fpr_limit, unique_fpr, unique_pro)

    within_limit = unique_fpr < fpr_limit

    curve_fpr = np.append(unique_fpr[within_limit], fpr_limit)
    curve_pro = np.append(unique_pro[within_limit], pro_at_limit)

    # Área normalizada: resultado entre 0 y 1.
    area = np.trapezoid(curve_pro, curve_fpr)

    return float(area / fpr_limit)


def calculate_segmentation_metrics(
    anomaly_maps,
    masks,
    threshold,
    fpr_limit=0.30,
):
    """Calcula AUPRO, IoU y Dice con un umbral ya seleccionado."""
    scores, targets = prepare_segmentation_arrays(anomaly_maps, masks)

    predicted = scores >= threshold

    intersection = np.count_nonzero(predicted & targets)
    union = np.count_nonzero(predicted | targets)

    denominator = predicted.sum() + targets.sum()

    iou = intersection / union if union > 0 else 1.0
    dice = 2 * intersection / denominator if denominator > 0 else 1.0

    return {
        "aupro": calculate_aupro(scores, targets, fpr_limit=fpr_limit),
        "iou": float(iou),
        "dice": float(dice),
    }
