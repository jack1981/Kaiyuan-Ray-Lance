#!/usr/bin/env bash

set -euo pipefail

DRIVER_MAX_RESULT_SIZE=${DRIVER_MAX_RESULT_SIZE:-8G}
DRIVER_MEMORY=${DRIVER_MEMORY:-6G}
EXECUTOR_MEMORY=${EXECUTOR_MEMORY:-6G}
EXECUTOR_MEMORY_OVERHEAD=${EXECUTOR_MEMORY_OVERHEAD:-2G}
EXECUTOR_CORES=${EXECUTOR_CORES:-4}
EXECUTOR_INSTANCES=${EXECUTOR_INSTANCES:-1}
SPARK_MASTER=${SPARK_MASTER:-local[*]}
LANCE_SPARK_PACKAGE=${LANCE_SPARK_PACKAGE:-com.lancedb:lance-spark-bundle-3.5_2.12:0.0.15}
LANCE_SPARK_EXTENSIONS=${LANCE_SPARK_EXTENSIONS:-1}

SCRIPT=${1:-}
CONFIG_FILE=${2:-}

if [ -z "$SCRIPT" ] || [ -z "$CONFIG_FILE" ]; then
  echo "Usage: bash script/run_local.sh <script.py> <config.yaml>" >&2
  exit 2
fi

if [ -z "${SPARK_HOME:-}" ]; then
  echo "SPARK_HOME is not set." >&2
  exit 2
fi

if [ ! -x "${SPARK_HOME}/bin/spark-submit" ]; then
  echo "spark-submit not found at ${SPARK_HOME}/bin/spark-submit" >&2
  exit 2
fi

# If JAVA_HOME is invalid, let Spark fall back to java on PATH.
if [ -n "${JAVA_HOME:-}" ] && [ ! -x "${JAVA_HOME}/bin/java" ]; then
  echo "Invalid JAVA_HOME=${JAVA_HOME}; falling back to java from PATH." >&2
  unset JAVA_HOME
fi

if ! command -v java >/dev/null 2>&1; then
  echo "java not found on PATH. Please install a JRE (Java 17 recommended)." >&2
  exit 2
fi

if [ -z "${JAVA_HOME:-}" ]; then
  java_bin=$(readlink -f "$(command -v java)")
  export JAVA_HOME="$(dirname "$(dirname "${java_bin}")")"
fi

export PYTHONPATH=./

submit_args=()

if [ "${LANCE_SPARK_EXTENSIONS}" = "1" ]; then
  if ls "${SPARK_HOME}"/jars/lance-spark-bundle*.jar >/dev/null 2>&1; then
    echo "Detected bundled lance-spark jar in ${SPARK_HOME}/jars."
  elif [ -n "${LANCE_SPARK_PACKAGE}" ]; then
    submit_args+=(--packages "${LANCE_SPARK_PACKAGE}")
  fi

  submit_args+=(
    --conf "spark.sql.catalog.lance=com.lancedb.lance.spark.LanceCatalog"
    --conf "spark.sql.extensions=com.lancedb.lance.spark.extensions.LanceSparkSessionExtensions"
    --conf "spark.sql.execution.arrow.maxRecordsPerBatch=4096"
    --conf "spark.sql.shuffle.partitions=32"
  )
fi

"${SPARK_HOME}/bin/spark-submit" \
  --master "${SPARK_MASTER}" \
  --conf "spark.driver.maxResultSize=${DRIVER_MAX_RESULT_SIZE}" \
  --conf "spark.driver.memory=${DRIVER_MEMORY}" \
  --conf "spark.executor.memory=${EXECUTOR_MEMORY}" \
  --conf "spark.executor.memoryOverhead=${EXECUTOR_MEMORY_OVERHEAD}" \
  --conf "spark.executor.cores=${EXECUTOR_CORES}" \
  --conf "spark.executor.instances=${EXECUTOR_INSTANCES}" \
  "${submit_args[@]}" \
  "$SCRIPT" \
  --config "$CONFIG_FILE"
