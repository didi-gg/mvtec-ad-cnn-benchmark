"""Ejecuta los experimentos en todas las categorías de MVTec AD.

Ejemplos:
    python scripts/run_experiments.py --dry-run
    python scripts/run_experiments.py
    python scripts/run_experiments.py --categories cable carpet
    python scripts/run_experiments.py --models patchcore
    python scripts/run_experiments.py --force
"""

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "config.yaml"
RESULTS_ROOT = PROJECT_ROOT / "results"


def load_configuration():
    """Lee la configuración general del proyecto."""
    with CONFIG_PATH.open(encoding="utf-8") as file:
        return yaml.safe_load(file)


def experiment_is_complete(model_name, category, config):
    """Comprueba que las dos variantes tengan modelo y métricas."""
    variants = config["models"][model_name]["variants"]

    for variant_name in variants:
        variant_dir = (
            PROJECT_ROOT
            / config["results"]["runs"]
            / model_name
            / category
            / variant_name
        )

        model_path = variant_dir / "model.pt"
        metrics_path = variant_dir / "metrics.json"

        if not model_path.is_file() or not metrics_path.is_file():
            return False

        try:
            with metrics_path.open(encoding="utf-8") as file:
                results = json.load(file)

            if (
                results["model"] != model_name
                or results["category"] != category
                or results["variant"] != variant_name
                or "detection" not in results["validation"]
                or "segmentation" not in results["validation"]
            ):
                return False

        except (OSError, ValueError, KeyError, TypeError):
            return False

    return True


def execute_experiment(model_name, category):
    """Ejecuta el pipeline existente en otro proceso Python."""
    command = [
        sys.executable,
        "-m",
        "src.experiment",
        "--category",
        category,
        "--model",
        model_name,
    ]

    start = time.perf_counter()

    result = subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        check=False,
    )

    return result.returncode, time.perf_counter() - start


def record_failed_experiment(model_name, category, error):
    """Registra fallos para facilitar la reanudación."""
    log_path = RESULTS_ROOT / "final" / "experiment_errors.jsonl"
    log_path.parent.mkdir(parents=True, exist_ok=True)

    record = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "model": model_name,
        "category": category,
        "error": str(error),
    }

    with log_path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(record, ensure_ascii=False) + "\n")


def run_all_experiments(config, categories, models, force, dry_run):
    """Ejecuta experimentos pendientes y omite los ya completados."""
    completed = 0
    skipped = 0
    failed = 0

    total = len(categories) * len(models)
    current = 0

    for category in categories:
        for model_name in models:
            current += 1

            print(
                f"\n[{current}/{total}] {category} / {model_name}",
                flush=True,
            )

            if not force and experiment_is_complete(model_name, category, config):
                print("Omitido: las dos variantes ya están completas.")
                skipped += 1
                continue

            if dry_run:
                print("Pendiente de ejecución.")
                continue

            try:
                return_code, seconds = execute_experiment(model_name, category)

                if return_code == 0 and experiment_is_complete(
                    model_name, category, config
                ):
                    completed += 1
                    print(f"Completado en {seconds / 60:.1f} minutos.")
                else:
                    failed += 1
                    record_failed_experiment(
                        model_name,
                        category,
                        f"Código de salida: {return_code}",
                    )
                    print("Error: revisar salida y registro de fallos.")

            except OSError as error:
                failed += 1
                record_failed_experiment(model_name, category, error)
                print(f"Error: {error}")

    print("\nRESUMEN")
    print("Completados:", completed)
    print("Omitidos:", skipped)
    print("Fallidos:", failed)

    if dry_run:
        print("Simulación terminada: no se ejecutaron modelos.")

    return failed


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--categories",
        nargs="+",
        help="Categorías específicas; por defecto se usan todas.",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        choices=["autoencoder", "patchcore", "dual_unet"],
        help="Modelos específicos; por defecto se usan todos.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Muestra las ejecuciones pendientes sin entrenar.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Reejecuta incluso si existen resultados completos.",
    )

    args = parser.parse_args()
    config = load_configuration()

    categories = args.categories or config["data"]["categories"]
    models = args.models or list(config["models"])

    unknown_categories = set(categories) - set(config["data"]["categories"])

    if unknown_categories:
        parser.error(f"Categorías desconocidas: {sorted(unknown_categories)}")

    failed = run_all_experiments(
        config=config,
        categories=categories,
        models=models,
        force=args.force,
        dry_run=args.dry_run,
    )

    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
