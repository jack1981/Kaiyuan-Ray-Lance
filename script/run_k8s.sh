#!/usr/bin/env bash

set -euo pipefail

SCRIPT=${1:-}
CONFIG_FILE=${2:-}

if [ -z "${SCRIPT}" ] || [ -z "${CONFIG_FILE}" ]; then
  echo "Usage: bash script/run_k8s.sh <script.py> <config.yaml>" >&2
  exit 2
fi

if ! command -v kubectl >/dev/null 2>&1; then
  echo "kubectl is required to submit Spark jobs to Kubernetes." >&2
  exit 2
fi

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if [ ! -f "${repo_root}/${SCRIPT}" ] && [ ! -f "${SCRIPT}" ]; then
  echo "Script not found: ${SCRIPT}" >&2
  exit 2
fi

if [ ! -f "${repo_root}/${CONFIG_FILE}" ] && [ ! -f "${CONFIG_FILE}" ]; then
  echo "Config not found: ${CONFIG_FILE}" >&2
  exit 2
fi

script_in_container="${SCRIPT}"
config_in_container="${CONFIG_FILE}"

if [[ "${SCRIPT}" = /* ]]; then
  if [[ "${SCRIPT}" == "${repo_root}/"* ]]; then
    script_in_container="/workspace/${SCRIPT#${repo_root}/}"
  else
    echo "Absolute script path must be under repository root: ${repo_root}" >&2
    exit 2
  fi
fi

if [[ "${CONFIG_FILE}" = /* ]]; then
  if [[ "${CONFIG_FILE}" == "${repo_root}/"* ]]; then
    config_in_container="/workspace/${CONFIG_FILE#${repo_root}/}"
  else
    echo "Absolute config path must be under repository root: ${repo_root}" >&2
    exit 2
  fi
fi

NAMESPACE=${K8S_NAMESPACE:-kaiyuan-spark}
SERVICE_ACCOUNT=${K8S_SERVICE_ACCOUNT:-spark}
SPARK_K8S_IMAGE=${SPARK_K8S_IMAGE:-kaiyuan-spark-app:latest}
SPARK_K8S_IMAGE_PULL_POLICY=${SPARK_K8S_IMAGE_PULL_POLICY:-IfNotPresent}
WAIT_TIMEOUT=${K8S_WAIT_TIMEOUT:-3600s}
DELETE_SUBMIT_JOB=${K8S_DELETE_SUBMIT_JOB:-1}

raw_job_name=${K8S_SUBMIT_JOB_NAME:-spark-submit-$(date +%s)-$RANDOM}
JOB_NAME=$(echo "${raw_job_name}" | tr '[:upper:]' '[:lower:]' | tr -c 'a-z0-9-' '-')
JOB_NAME=${JOB_NAME#-}
JOB_NAME=${JOB_NAME%-}
JOB_NAME=${JOB_NAME:0:63}
JOB_NAME=${JOB_NAME%-}

if [ -z "${JOB_NAME}" ]; then
  JOB_NAME="spark-submit-$(date +%s)"
fi

echo "Submitting Spark-on-K8s job '${JOB_NAME}' in namespace '${NAMESPACE}'"

kubectl -n "${NAMESPACE}" delete job "${JOB_NAME}" --ignore-not-found >/dev/null 2>&1 || true

cat <<EOF_YAML | kubectl apply -f -
apiVersion: batch/v1
kind: Job
metadata:
  name: ${JOB_NAME}
  namespace: ${NAMESPACE}
spec:
  backoffLimit: 0
  template:
    spec:
      serviceAccountName: ${SERVICE_ACCOUNT}
      restartPolicy: Never
      containers:
        - name: spark-submit
          image: ${SPARK_K8S_IMAGE}
          imagePullPolicy: ${SPARK_K8S_IMAGE_PULL_POLICY}
          workingDir: /workspace
          command: ["/bin/bash", "-lc"]
          args: ["bash script/run_k8s_submit.sh ${script_in_container} ${config_in_container}"]
          env:
            - name: K8S_NAMESPACE
              value: "${NAMESPACE}"
            - name: K8S_SERVICE_ACCOUNT
              value: "${SERVICE_ACCOUNT}"
            - name: SPARK_K8S_IMAGE
              value: "${SPARK_K8S_IMAGE}"
            - name: SPARK_K8S_IMAGE_PULL_POLICY
              value: "${SPARK_K8S_IMAGE_PULL_POLICY}"
            - name: SPARK_K8S_EXECUTOR_DELETE_ON_TERMINATION
              value: "${SPARK_K8S_EXECUTOR_DELETE_ON_TERMINATION:-true}"
            - name: SPARK_K8S_DRIVER_DELETE_ON_TERMINATION
              value: "${SPARK_K8S_DRIVER_DELETE_ON_TERMINATION:-true}"
            - name: SPARK_K8S_MASTER
              value: "${SPARK_K8S_MASTER:-k8s://https://kubernetes.default.svc}"
            - name: K8S_EXECUTOR_INSTANCES
              value: "${K8S_EXECUTOR_INSTANCES:-2}"
            - name: DRIVER_MAX_RESULT_SIZE
              value: "${DRIVER_MAX_RESULT_SIZE:-4G}"
            - name: DRIVER_MEMORY
              value: "${DRIVER_MEMORY:-3G}"
            - name: EXECUTOR_MEMORY
              value: "${EXECUTOR_MEMORY:-2G}"
            - name: EXECUTOR_MEMORY_OVERHEAD
              value: "${EXECUTOR_MEMORY_OVERHEAD:-1G}"
            - name: EXECUTOR_CORES
              value: "${EXECUTOR_CORES:-2}"
            - name: LANCE_SPARK_PACKAGE
              value: "${LANCE_SPARK_PACKAGE:-com.lancedb:lance-spark-bundle-3.5_2.12:0.0.15}"
            - name: LANCE_SPARK_EXTENSIONS
              value: "${LANCE_SPARK_EXTENSIONS:-1}"
            - name: MINIO_ENDPOINT
              value: "${MINIO_ENDPOINT:-http://minio.kaiyuan-spark.svc.cluster.local:9000}"
            - name: MINIO_ACCESS_KEY
              value: "${MINIO_ACCESS_KEY:-minio}"
            - name: MINIO_SECRET_KEY
              value: "${MINIO_SECRET_KEY:-minio123}"
            - name: MINIO_REGION
              value: "${MINIO_REGION:-us-east-1}"
            - name: MINIO_BUCKET
              value: "${MINIO_BUCKET:-kaiyuan-spark}"
            - name: SPARK_FILE_UPLOAD_PATH
              value: "${SPARK_FILE_UPLOAD_PATH:-s3a://kaiyuan-spark/spark-upload}"
            - name: SPARK_MODEL_FILES
              value: "${SPARK_MODEL_FILES:-s3a://kaiyuan-spark/models/fasttext_hq.bin,s3a://kaiyuan-spark/models/fasttext_mmlu.bin}"
            - name: SPARK_MODEL_ARCHIVES
              value: "${SPARK_MODEL_ARCHIVES:-s3a://kaiyuan-spark/models/tiny_seq_classifier.zip#tiny_seq_classifier}"
            - name: K8S_CONFIG_REWRITE
              value: "${K8S_CONFIG_REWRITE:-1}"
            - name: K8S_SAMPLE_PREFIX
              value: "${K8S_SAMPLE_PREFIX:-s3a://kaiyuan-spark/sample}"
            - name: K8S_OUTPUT_PREFIX
              value: "${K8S_OUTPUT_PREFIX:-s3a://kaiyuan-spark/output}"
            - name: SPARK_EVENT_LOG_ENABLED
              value: "${SPARK_EVENT_LOG_ENABLED:-true}"
            - name: SPARK_EVENT_LOG_DIR
              value: "${SPARK_EVENT_LOG_DIR:-s3a://kaiyuan-spark/spark-events}"
EOF_YAML

if ! kubectl -n "${NAMESPACE}" wait --for=condition=complete "job/${JOB_NAME}" --timeout="${WAIT_TIMEOUT}"; then
  echo "Spark submit job failed or timed out: ${JOB_NAME}" >&2
  kubectl -n "${NAMESPACE}" describe "job/${JOB_NAME}" || true
  kubectl -n "${NAMESPACE}" logs "job/${JOB_NAME}" --all-containers=true || true
  exit 1
fi

tmp_log_file="$(mktemp -t spark-k8s-submit-log.XXXXXX)"
kubectl -n "${NAMESPACE}" logs "job/${JOB_NAME}" --all-containers=true | tee "${tmp_log_file}"

if command -v rg >/dev/null 2>&1; then
  failure_detected=0
  if rg -q "phase: Failed|termination reason: StartError|Application status for .*\\(phase: Failed\\)" "${tmp_log_file}"; then
    failure_detected=1
  fi
else
  failure_detected=0
  if grep -Eq "phase: Failed|termination reason: StartError|Application status for .*\\(phase: Failed\\)" "${tmp_log_file}"; then
    failure_detected=1
  fi
fi

if [ "${failure_detected}" = "1" ]; then
  echo "Spark application reported failed status. See logs above." >&2
  rm -f "${tmp_log_file}"
  exit 1
fi

rm -f "${tmp_log_file}"

if [ "${DELETE_SUBMIT_JOB}" = "1" ]; then
  kubectl -n "${NAMESPACE}" delete job "${JOB_NAME}" --ignore-not-found >/dev/null 2>&1 || true
fi
