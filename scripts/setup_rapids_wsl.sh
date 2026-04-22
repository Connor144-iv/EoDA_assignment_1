#!/usr/bin/env bash
set -euo pipefail

log() {
  printf '\n==> %s\n' "$1"
}

die() {
  printf 'ERROR: %s\n' "$1" >&2
  exit 1
}

have() {
  command -v "$1" >/dev/null 2>&1
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SOURCE_REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
TARGET_REPO_DIR="$HOME/projects/ml_lab/eng_of_data_analysis"
MINIFORGE_DIR="${MINIFORGE_DIR:-$HOME/miniforge3}"
ENV_NAME="${ENV_NAME:-rapids-eoda}"
MINIFORGE_INSTALLER="/tmp/Miniforge3-$(uname)-$(uname -m).sh"

if ! have sudo; then
  die "sudo is required inside Ubuntu to install system packages."
fi

log "Installing base Ubuntu packages"
sudo apt-get update
sudo apt-get install -y curl ca-certificates bzip2 git rsync

if [[ ! -x "$MINIFORGE_DIR/bin/conda" ]]; then
  log "Installing Miniforge"
  curl -L -o "$MINIFORGE_INSTALLER" "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-$(uname)-$(uname -m).sh"
  bash "$MINIFORGE_INSTALLER" -b -p "$MINIFORGE_DIR"
fi

log "Initializing Conda"
source "$MINIFORGE_DIR/etc/profile.d/conda.sh"
conda config --set solver libmamba
conda config --set channel_priority strict

log "Creating or updating the RAPIDS environment"
CONDA_ARGS=(
  --override-channels
  -c rapidsai
  -c conda-forge
  -c nodefaults
  python=3.13
  cudf
  cuml
  cupy
  pandas
  numpy
  matplotlib
  scikit-learn
  jupyterlab
  ipykernel
  cuda-version=13.1
)

if conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
  conda install -y -n "$ENV_NAME" "${CONDA_ARGS[@]}"
else
  conda create -y -n "$ENV_NAME" "${CONDA_ARGS[@]}"
fi

log "Registering the Jupyter kernel"
conda run -n "$ENV_NAME" python -m ipykernel install --user --name "$ENV_NAME" --display-name "Python ($ENV_NAME)"

log "Copying the repo into the WSL filesystem"
mkdir -p "$(dirname "$TARGET_REPO_DIR")"
if [[ "$SOURCE_REPO_DIR" != "$TARGET_REPO_DIR" ]]; then
  rsync -a "$SOURCE_REPO_DIR/" "$TARGET_REPO_DIR/"
fi

log "Validating package channels"
if conda list -n "$ENV_NAME" --show-channel-urls | grep -q 'defaults'; then
  die "The RAPIDS environment contains packages from the defaults channel."
fi

log "Checking GPU visibility inside WSL"
if have nvidia-smi; then
  nvidia-smi
elif [[ -x /usr/lib/wsl/lib/nvidia-smi ]]; then
  /usr/lib/wsl/lib/nvidia-smi
else
  die "nvidia-smi is not available inside WSL."
fi

log "Running RAPIDS validation"
conda run -n "$ENV_NAME" python --version
conda run -n "$ENV_NAME" python -c "import cupy as cp; print('device_count=', cp.cuda.runtime.getDeviceCount())"
conda run -n "$ENV_NAME" python -c "import cudf, cupy, cuml.cluster; print(cudf.Series([1, 2, 3]))"

log "Running notebook import and cuML smoke test"
cd "$TARGET_REPO_DIR"
conda run -n "$ENV_NAME" python - <<'PY'
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn.cluster
import cudf
import cupy as cp
from cuml.cluster import KMeans

assert cp.cuda.runtime.getDeviceCount() >= 1

gdf = cudf.DataFrame(
    {
        "x": cp.asarray([0.0, 0.1, 9.0, 9.1], dtype=cp.float32),
        "y": cp.asarray([0.0, 0.1, 9.0, 9.1], dtype=cp.float32),
    }
)
model = KMeans(n_clusters=2, max_iter=20, init="k-means++", random_state=0)
labels = model.fit_predict(gdf[["x", "y"]])
print("labels=", labels.to_pandas().tolist())
print("repo=", "class_code/EoDA_lecture_3.ipynb")
PY

log "Setup complete"
printf 'WSL repo path: %s\n' "$TARGET_REPO_DIR"
printf 'Activate env with: conda activate %s\n' "$ENV_NAME"
printf 'Open in VS Code with: cd %s && code .\n' "$TARGET_REPO_DIR"
