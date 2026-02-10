#!/usr/bin/env bash

set -euo pipefail

SCRIPT=${1:-}
CONFIG_FILE=${2:-}

if [ -z "${SCRIPT}" ] || [ -z "${CONFIG_FILE}" ]; then
  echo "Usage: bash script/run_local.sh <script.py> <config.yaml>" >&2
  exit 2
fi

if [ ! -f "${SCRIPT}" ]; then
  echo "Script not found: ${SCRIPT}" >&2
  exit 2
fi

if [ ! -f "${CONFIG_FILE}" ]; then
  echo "Config not found: ${CONFIG_FILE}" >&2
  exit 2
fi

export PYTHONPATH=./

args=(
  "${SCRIPT}"
  --config "${CONFIG_FILE}"
  --mode local
)

if [ -n "${RAY_ADDRESS:-}" ]; then
  args+=(--ray-address "${RAY_ADDRESS}")
fi

python "${args[@]}"
