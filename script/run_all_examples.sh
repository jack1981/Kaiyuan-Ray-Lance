#!/usr/bin/env bash
set -euo pipefail

CONFIG_DIR=${1:-example}

required_files=(
  "/data/sample/pcmind_kaiyuan_2b_sample.parquet"
  "/data/sample/scored_input.parquet"
  "/data/sample/fineweb_chinese.parquet"
  "/data/sample/dclm_subset.parquet"
  "/data/sample/dclm_subset_dedup.parquet"
  "/data/models/fasttext_hq.bin"
  "/data/models/fasttext_mmlu.bin"
  "/data/models/tiny_seq_classifier/config.json"
)

for file in "${required_files[@]}"; do
  if [ ! -f "$file" ]; then
    echo "Missing required asset: $file"
    echo "Run: make prepare-examples"
    exit 2
  fi
done

for cfg in "${CONFIG_DIR}"/*.yaml; do
  echo "=============================="
  echo "Running config: ${cfg}"
  echo "=============================="
  bash script/run_local.sh main.py "${cfg}"
done

echo "All example configs completed successfully."
