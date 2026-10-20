"""
Preparación de datos y particiones reproducibles de MVTec AD.

Objetivo:
    Crear particiones de entrenamiento (80 %) y validación (20 %)
    utilizando exclusivamente imágenes normales del train oficial.

Ejecución:
    python scripts/prepare_data.py
"""

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

# ============================================================
# 1. Configuración
# ============================================================
PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data" / "mvtec_anomaly_detection"
RESULTS_DIR = PROJECT_ROOT / "results" / "splits"

SEED = 1926
TRAIN_FRACTION = 0.80
VAL_FRACTION = 0.20

CATEGORIES = [
    "bottle",
    "cable",
    "capsule",
    "carpet",
    "grid",
    "hazelnut",
    "leather",
    "metal_nut",
    "pill",
    "screw",
    "tile",
    "toothbrush",
    "transistor",
    "wood",
    "zipper",
]


# ============================================================
# 2. Localizar el dataset
# ============================================================
def resolve_dataset_root() -> Path:
    if (DATA_DIR / "bottle" / "train" / "good").is_dir():
        return DATA_DIR

    candidates = list(DATA_DIR.glob("*/bottle/train/good"))

    if len(candidates) == 1:
        return candidates[0].parents[2]

    raise FileNotFoundError(f"No se encontró MVTec AD en {DATA_DIR}")


# ============================================================
# 3. Verificar duplicados exactos
# ============================================================
def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as file:
        for chunk in iter(
            lambda: file.read(1024 * 1024),
            b"",
        ):
            digest.update(chunk)

    return digest.hexdigest()


def audit_duplicates(dataset_root: Path) -> None:
    records = []

    for category in CATEGORIES:
        for split in ("train", "test"):
            split_dir = dataset_root / category / split

            if not split_dir.is_dir():
                raise FileNotFoundError(split_dir)

            for image_path in sorted(split_dir.glob("*/*.png")):
                records.append(
                    {
                        "category": category,
                        "split": split,
                        "path": str(image_path.relative_to(dataset_root)),
                        "sha256": file_sha256(image_path),
                    }
                )

    df = pd.DataFrame(records)

    if len(df) != 5354:
        raise ValueError(f"Se esperaban 5354 imágenes; se encontraron {len(df)}.")

    duplicates = df[df.duplicated("sha256", keep=False)]

    train_hashes = set(df.loc[df["split"] == "train", "sha256"])
    test_hashes = set(df.loc[df["split"] == "test", "sha256"])
    shared_hashes = train_hashes & test_hashes

    print(f"Imágenes analizadas: {len(df)}")
    print(f"Archivos involucrados en duplicados: {len(duplicates)}")
    print(f"Hashes compartidos train/test: {len(shared_hashes)}")

    if not duplicates.empty:
        raise RuntimeError(
            "Se encontraron imágenes duplicadas. Revisar antes de crear particiones."
        )


# ============================================================
# 4. Generar particiones
# ============================================================
def create_split(
    category: str,
    dataset_root: Path,
) -> dict:
    """
    Genera un split 80/20 reproducible para una categoría.

    Las imágenes se mantienen en su ubicación original.
    Se guardan únicamente sus rutas relativas.
    """

    train_dir = dataset_root / category / "train" / "good"

    images = sorted(train_dir.glob("*.png"))

    if len(images) < 2:
        raise ValueError(f"{category}: imágenes insuficientes.")

    rng = np.random.default_rng(SEED)

    indices = rng.permutation(len(images))

    n_validation = max(
        1,
        round(len(images) * VAL_FRACTION),
    )

    validation_indices = indices[:n_validation]
    training_indices = indices[n_validation:]

    training_paths = sorted(
        str(images[i].relative_to(dataset_root)) for i in training_indices
    )

    validation_paths = sorted(
        str(images[i].relative_to(dataset_root)) for i in validation_indices
    )

    # Verificar ausencia de solapamiento
    assert set(training_paths).isdisjoint(validation_paths)

    # Verificar conservación de las imágenes
    assert len(training_paths) + len(validation_paths) == len(images)

    return {
        "category": category,
        "seed": SEED,
        "train_fraction": TRAIN_FRACTION,
        "validation_fraction": VAL_FRACTION,
        "train": training_paths,
        "validation": validation_paths,
    }


# ============================================================
# 5. Preparación y persistencia
# ============================================================
def main() -> None:

    dataset_root = resolve_dataset_root()

    print("\n=== Preparación de MVTec AD ===")
    print("Dataset:", dataset_root)
    print("Semilla:", SEED)
    print("Partición: 80 % entrenamiento / 20 % validación")

    # Verificar estructura completa
    for category in CATEGORIES:
        for directory in ("train", "test", "ground_truth"):
            path = dataset_root / category / directory

            if not path.is_dir():
                raise FileNotFoundError(path)

    # Auditoría de integridad
    print("\n=== Auditoría de duplicados ===")
    audit_duplicates(dataset_root)

    # Evitar modificar silenciosamente un split ya utilizado
    if RESULTS_DIR.exists() and list(RESULTS_DIR.glob("*.json")):
        raise FileExistsError(
            f"Ya existen particiones en {RESULTS_DIR}. "
            "No se sobrescribirán automáticamente."
        )

    manifests = {}
    summary = []

    # Generar todas las particiones antes de escribir archivos
    for category in CATEGORIES:
        manifest = create_split(category, dataset_root)
        manifests[category] = manifest

        summary.append(
            {
                "category": category,
                "train": len(manifest["train"]),
                "validation": len(manifest["validation"]),
                "total": (len(manifest["train"]) + len(manifest["validation"])),
            }
        )

    df_summary = pd.DataFrame(summary)

    # Verificaciones globales
    assert len(df_summary) == 15
    assert df_summary["total"].sum() == 3629
    assert df_summary["train"].sum() > 0
    assert df_summary["validation"].sum() > 0

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Guardar manifiestos JSON
    for category, manifest in manifests.items():
        output_path = RESULTS_DIR / f"{category}.json"

        output_path.write_text(
            json.dumps(manifest, indent=2),
            encoding="utf-8",
        )

    # Guardar resumen CSV
    df_summary["validation_percentage"] = (
        df_summary["validation"] / df_summary["total"] * 100
    )

    df_summary.to_csv(
        RESULTS_DIR / "summary.csv",
        index=False,
    )

    print("\n=== Particiones generadas ===")
    print(df_summary.to_string(index=False))

    print("\n=== Totales ===")
    print(df_summary[["train", "validation", "total"]].sum().to_string())

    print("\nVerificaciones: OK")
    print("Test oficial: sin modificaciones")
    print("Resultados:", RESULTS_DIR)


if __name__ == "__main__":
    main()
