#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="mvtec-cnn"

cd "$(dirname "${BASH_SOURCE[0]}")"

# 1. Verificar Conda
if ! command -v conda >/dev/null 2>&1; then
    echo "ERROR: Instala Conda o Miniforge."
    exit 1
fi

eval "$(conda shell.bash hook)"

# 2. Crear entorno si no existe
if ! conda env list | awk -v name="$ENV_NAME" \
    '$1 == name {found=1} END {exit !found}'; then
    conda create -n "$ENV_NAME" python=3.11 pip -y
fi

conda activate "$ENV_NAME"

echo "Entorno: $CONDA_DEFAULT_ENV"
python --version

# 3. Actualizar pip
python -m pip install --upgrade pip

# 4. Dependencias obligatorias
PACKAGES=(
    torch
    torchvision
    numpy
    pandas
    scipy
    scikit-learn
    scikit-image
    opencv-python
    Pillow
    matplotlib
    seaborn
    plotly
    PyYAML
    psutil
    torchinfo
    tqdm
    streamlit
    jupyterlab
    ipykernel
    pytest
)

# 5. Respetar las versiones ya registradas
if [ -f requirements.txt ]; then
    python -m pip install -r requirements.txt
fi

# 6. Instalar las dependencias faltantes sin actualizar las existentes
python -m pip install "${PACKAGES[@]}"

# 7. Actualizar requirements.txt con versiones exactas
python - <<'PY'
from importlib.metadata import version
from pathlib import Path

packages = [
    "torch", "torchvision", "numpy", "pandas",
    "scipy", "scikit-learn", "scikit-image",
    "opencv-python", "Pillow", "matplotlib",
    "seaborn", "plotly", "PyYAML", "psutil",
    "torchinfo", "tqdm", "streamlit",
    "jupyterlab", "ipykernel", "pytest"
]

Path("requirements.txt").write_text(
    "\n".join(f"{p}=={version(p)}" for p in packages) + "\n"
)

print("requirements.txt actualizado correctamente.")
PY

# 8. Registrar kernel Jupyter
python -m ipykernel install --user \
    --name "$ENV_NAME" \
    --display-name "Python (MVTec CNN)"

# 9. Verificar dependencias
python -m pip check

# 10. Verificar imports y MPS
python - <<'PY'
import torch
import torchvision
import cv2
import seaborn
import plotly
import streamlit

print("\n=== Verificación del entorno ===")
print("PyTorch:", torch.__version__)
print("Torchvision:", torchvision.__version__)
print("OpenCV:", cv2.__version__)
print("Seaborn:", seaborn.__version__)
print("Plotly:", plotly.__version__)
print("Streamlit:", streamlit.__version__)

mps = torch.backends.mps.is_available()
device = "mps" if mps else "cpu"

x = torch.randn(256, 256, device=device)
y = x @ x

if mps:
    torch.mps.synchronize()

print("MPS disponible:", mps)
print("Prueba de operaciones: OK")
print("Dispositivo:", device)
PY

echo ""
echo "Instalación finalizada."
echo "Activar: conda activate $ENV_NAME"
echo "Jupyter: jupyter lab"
echo "Streamlit: streamlit run app.py"