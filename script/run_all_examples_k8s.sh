#!/usr/bin/env bash

set -euo pipefail

CONFIG_DIR=${1:-example}

if [ ! -d "${CONFIG_DIR}" ]; then
  echo "Config directory not found: ${CONFIG_DIR}" >&2
  exit 2
fi

for cfg in "${CONFIG_DIR}"/*.yaml; do
  echo "=============================="
  echo "Running K8s config: ${cfg}"
  echo "=============================="
  bash script/run_k8s.sh main.py "${cfg}"
done

echo "All K8s example configs completed successfully."
