#!/usr/bin/env bash

set -euo pipefail

SCRIPT=${1:-}
CONFIG_FILE=${2:-}

if [ -z "${SCRIPT}" ] || [ -z "${CONFIG_FILE}" ]; then
  echo "Usage: bash script/run_k8s_submit.sh <script.py> <config.yaml>" >&2
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

if [ -z "${SPARK_HOME:-}" ]; then
  echo "SPARK_HOME is not set." >&2
  exit 2
fi

if [ ! -x "${SPARK_HOME}/bin/spark-submit" ]; then
  echo "spark-submit not found at ${SPARK_HOME}/bin/spark-submit" >&2
  exit 2
fi

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

export PYTHONPATH=/workspace

DRIVER_MAX_RESULT_SIZE=${DRIVER_MAX_RESULT_SIZE:-4G}
DRIVER_MEMORY=${DRIVER_MEMORY:-3G}
EXECUTOR_MEMORY=${EXECUTOR_MEMORY:-2G}
EXECUTOR_MEMORY_OVERHEAD=${EXECUTOR_MEMORY_OVERHEAD:-1G}
EXECUTOR_CORES=${EXECUTOR_CORES:-2}
EXECUTOR_INSTANCES=${K8S_EXECUTOR_INSTANCES:-2}
SPARK_K8S_MASTER=${SPARK_K8S_MASTER:-k8s://https://kubernetes.default.svc}
K8S_NAMESPACE=${K8S_NAMESPACE:-kaiyuan-spark}
K8S_SERVICE_ACCOUNT=${K8S_SERVICE_ACCOUNT:-spark}
SPARK_K8S_IMAGE=${SPARK_K8S_IMAGE:-kaiyuan-spark-app:latest}
SPARK_K8S_IMAGE_PULL_POLICY=${SPARK_K8S_IMAGE_PULL_POLICY:-IfNotPresent}
SPARK_K8S_EXECUTOR_DELETE_ON_TERMINATION=${SPARK_K8S_EXECUTOR_DELETE_ON_TERMINATION:-true}
SPARK_K8S_DRIVER_DELETE_ON_TERMINATION=${SPARK_K8S_DRIVER_DELETE_ON_TERMINATION:-true}
K8S_EXECUTOR_POD_TEMPLATE=${K8S_EXECUTOR_POD_TEMPLATE:-/workspace/k8s/pod-templates/executor.yaml}
LANCE_SPARK_PACKAGE=${LANCE_SPARK_PACKAGE:-com.lancedb:lance-spark-bundle-3.5_2.12:0.0.15}
LANCE_SPARK_EXTENSIONS=${LANCE_SPARK_EXTENSIONS:-1}
K8S_CONFIG_REWRITE=${K8S_CONFIG_REWRITE:-1}

MINIO_ENDPOINT=${MINIO_ENDPOINT:-http://minio.kaiyuan-spark.svc.cluster.local:9000}
MINIO_ACCESS_KEY=${MINIO_ACCESS_KEY:-minio}
MINIO_SECRET_KEY=${MINIO_SECRET_KEY:-minio123}
MINIO_REGION=${MINIO_REGION:-us-east-1}
MINIO_BUCKET=${MINIO_BUCKET:-kaiyuan-spark}
SPARK_FILE_UPLOAD_PATH=${SPARK_FILE_UPLOAD_PATH:-s3a://${MINIO_BUCKET}/spark-upload}
K8S_SAMPLE_PREFIX=${K8S_SAMPLE_PREFIX:-s3a://${MINIO_BUCKET}/sample}
K8S_OUTPUT_PREFIX=${K8S_OUTPUT_PREFIX:-s3a://${MINIO_BUCKET}/output}
SPARK_EVENT_LOG_ENABLED=${SPARK_EVENT_LOG_ENABLED:-true}
SPARK_EVENT_LOG_DIR=${SPARK_EVENT_LOG_DIR:-s3a://${MINIO_BUCKET}/spark-events}

LANCE_AWS_ALLOW_HTTP=false
if [[ "${MINIO_ENDPOINT}" == http://* ]]; then
  LANCE_AWS_ALLOW_HTTP=true
fi

SPARK_MODEL_FILES=${SPARK_MODEL_FILES:-s3a://${MINIO_BUCKET}/models/fasttext_hq.bin,s3a://${MINIO_BUCKET}/models/fasttext_mmlu.bin}
SPARK_MODEL_ARCHIVES=${SPARK_MODEL_ARCHIVES:-s3a://${MINIO_BUCKET}/models/tiny_seq_classifier.zip#tiny_seq_classifier}

sanitize_spark_files() {
  local files_csv="${1:-}"
  local had_alias=0
  local sanitized_parts=()
  local part

  if [ -z "${files_csv}" ]; then
    return 0
  fi

  local IFS=','
  # shellcheck disable=SC2206
  local parts=(${files_csv})
  for part in "${parts[@]}"; do
    if [[ "${part}" == *"#"* ]]; then
      had_alias=1
      sanitized_parts+=("${part%%#*}")
    else
      sanitized_parts+=("${part}")
    fi
  done

  if [ "${had_alias}" = "1" ]; then
    echo "Detected '#alias' entries in SPARK_MODEL_FILES. Stripping aliases to avoid Spark 'URI has a fragment component' failure on Kubernetes." >&2
  fi

  (
    local IFS=','
    echo "${sanitized_parts[*]}"
  )
}

CONFIG_TO_SUBMIT=${CONFIG_FILE}
if [ "${K8S_CONFIG_REWRITE}" = "1" ]; then
  rendered_config="/tmp/k8s-$(basename "${CONFIG_FILE}")"
  python - "${CONFIG_FILE}" "${rendered_config}" <<'PY'
import os
import sys
import yaml

src, dst = sys.argv[1], sys.argv[2]

bucket = os.environ.get("MINIO_BUCKET", "kaiyuan-spark")
sample_prefix = os.environ.get("K8S_SAMPLE_PREFIX", f"s3a://{bucket}/sample").rstrip("/")
output_prefix = os.environ.get("K8S_OUTPUT_PREFIX", f"s3a://{bucket}/output").rstrip("/")

model_map = {
    "/data/models/fasttext_hq.bin": "fasttext_hq.bin",
    "/data/models/fasttext_mmlu.bin": "fasttext_mmlu.bin",
    "/data/models/tiny_seq_classifier": "tiny_seq_classifier",
}


def rewrite_string(value: str) -> str:
    if value in model_map:
        return model_map[value]
    if value.startswith("/data/sample/"):
        return sample_prefix + "/" + value[len("/data/sample/") :]
    if value.startswith("/data/output/"):
        return output_prefix + "/" + value[len("/data/output/") :]
    if value.startswith("file:/data/sample/"):
        return sample_prefix + "/" + value[len("file:/data/sample/") :]
    if value.startswith("file:/data/output/"):
        return output_prefix + "/" + value[len("file:/data/output/") :]
    return value


def walk(node):
    if isinstance(node, dict):
        return {k: walk(v) for k, v in node.items()}
    if isinstance(node, list):
        return [walk(v) for v in node]
    if isinstance(node, str):
        return rewrite_string(node)
    return node

with open(src, "r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)

rewritten = walk(cfg)
with open(dst, "w", encoding="utf-8") as f:
    yaml.safe_dump(rewritten, f, sort_keys=False)

print(f"Rendered K8s config: {dst}")
PY
  CONFIG_TO_SUBMIT="${rendered_config}"
fi

submit_args=()

# Ship local Python package code to driver/executors in cluster mode.
if [ -d "/workspace/datafiner" ]; then
  pyfiles_archive="/tmp/datafiner-pyfiles.zip"
  python - "${pyfiles_archive}" <<'PY'
import os
import sys
import zipfile

dst = sys.argv[1]
src_root = "/workspace"
pkg_root = os.path.join(src_root, "datafiner")

with zipfile.ZipFile(dst, "w", compression=zipfile.ZIP_DEFLATED) as zf:
    for root, _, files in os.walk(pkg_root):
        for name in files:
            full_path = os.path.join(root, name)
            rel_path = os.path.relpath(full_path, src_root)
            zf.write(full_path, rel_path)
PY
  submit_args+=(--py-files "${pyfiles_archive}")
fi

if [ "${LANCE_SPARK_EXTENSIONS}" = "1" ]; then
  if ls "${SPARK_HOME}"/jars/lance-spark-bundle*.jar >/dev/null 2>&1; then
    echo "Detected bundled lance-spark jar in ${SPARK_HOME}/jars."
  elif [ -n "${LANCE_SPARK_PACKAGE}" ]; then
    submit_args+=(--packages "${LANCE_SPARK_PACKAGE}")
  fi

  submit_args+=(
    --conf "spark.sql.catalog.lance=com.lancedb.lance.spark.LanceCatalog"
    --conf "spark.sql.extensions=com.lancedb.lance.spark.extensions.LanceSparkSessionExtensions"
  )
fi

spark_files="${CONFIG_TO_SUBMIT}"
if [ -n "${SPARK_MODEL_FILES}" ]; then
  sanitized_model_files="$(sanitize_spark_files "${SPARK_MODEL_FILES}")"
  if [ -n "${sanitized_model_files}" ]; then
    spark_files="${spark_files},${sanitized_model_files}"
  fi
fi
submit_args+=(--files "${spark_files}")

if [ -n "${SPARK_MODEL_ARCHIVES}" ]; then
  submit_args+=(--archives "${SPARK_MODEL_ARCHIVES}")
fi

if [ -n "${K8S_EXECUTOR_POD_TEMPLATE:-}" ] && [ -f "${K8S_EXECUTOR_POD_TEMPLATE}" ]; then
  submit_args+=(--conf "spark.kubernetes.executor.podTemplateFile=${K8S_EXECUTOR_POD_TEMPLATE}")
fi

"${SPARK_HOME}/bin/spark-submit" \
  --master "${SPARK_K8S_MASTER}" \
  --deploy-mode cluster \
  --conf "spark.driver.maxResultSize=${DRIVER_MAX_RESULT_SIZE}" \
  --conf "spark.driver.memory=${DRIVER_MEMORY}" \
  --conf "spark.executor.memory=${EXECUTOR_MEMORY}" \
  --conf "spark.executor.memoryOverhead=${EXECUTOR_MEMORY_OVERHEAD}" \
  --conf "spark.executor.cores=${EXECUTOR_CORES}" \
  --conf "spark.executor.instances=${EXECUTOR_INSTANCES}" \
  --conf "spark.dynamicAllocation.enabled=false" \
  --conf "spark.kubernetes.namespace=${K8S_NAMESPACE}" \
  --conf "spark.kubernetes.authenticate.driver.serviceAccountName=${K8S_SERVICE_ACCOUNT}" \
  --conf "spark.kubernetes.container.image=${SPARK_K8S_IMAGE}" \
  --conf "spark.kubernetes.container.image.pullPolicy=${SPARK_K8S_IMAGE_PULL_POLICY}" \
  --conf "spark.kubernetes.executor.deleteOnTermination=${SPARK_K8S_EXECUTOR_DELETE_ON_TERMINATION}" \
  --conf "spark.kubernetes.driver.deleteOnTermination=${SPARK_K8S_DRIVER_DELETE_ON_TERMINATION}" \
  --conf "spark.kubernetes.file.upload.path=${SPARK_FILE_UPLOAD_PATH}" \
  --conf "spark.pyspark.python=python" \
  --conf "spark.pyspark.driver.python=python" \
  --conf "spark.executorEnv.PYSPARK_PYTHON=python" \
  --conf "spark.kubernetes.driverEnv.PYSPARK_PYTHON=python" \
  --conf "spark.sql.execution.arrow.maxRecordsPerBatch=4096" \
  --conf "spark.sql.shuffle.partitions=32" \
  --conf "spark.eventLog.enabled=${SPARK_EVENT_LOG_ENABLED}" \
  --conf "spark.eventLog.compress=true" \
  --conf "spark.eventLog.dir=${SPARK_EVENT_LOG_DIR}" \
  --conf "spark.hadoop.fs.s3a.impl=org.apache.hadoop.fs.s3a.S3AFileSystem" \
  --conf "spark.hadoop.fs.s3a.endpoint=${MINIO_ENDPOINT}" \
  --conf "spark.hadoop.fs.s3a.path.style.access=true" \
  --conf "spark.hadoop.fs.s3a.connection.ssl.enabled=false" \
  --conf "spark.hadoop.fs.s3a.access.key=${MINIO_ACCESS_KEY}" \
  --conf "spark.hadoop.fs.s3a.secret.key=${MINIO_SECRET_KEY}" \
  --conf "spark.hadoop.fs.s3a.aws.credentials.provider=org.apache.hadoop.fs.s3a.SimpleAWSCredentialsProvider" \
  --conf "spark.sql.catalog.lance.aws_access_key_id=${MINIO_ACCESS_KEY}" \
  --conf "spark.sql.catalog.lance.aws_secret_access_key=${MINIO_SECRET_KEY}" \
  --conf "spark.sql.catalog.lance.aws_endpoint=${MINIO_ENDPOINT}" \
  --conf "spark.sql.catalog.lance.aws_region=${MINIO_REGION}" \
  --conf "spark.sql.catalog.lance.aws_allow_http=${LANCE_AWS_ALLOW_HTTP}" \
  --conf "spark.sql.catalog.lance.aws_virtual_hosted_style_request=false" \
  "${submit_args[@]}" \
  "${SCRIPT}" \
  --config "$(basename "${CONFIG_TO_SUBMIT}")"
