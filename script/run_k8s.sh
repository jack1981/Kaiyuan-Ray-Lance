#!/usr/bin/env bash

set -euo pipefail

SCRIPT=${1:-}
CONFIG_FILE=${2:-}

if [ -z "${SCRIPT}" ] || [ -z "${CONFIG_FILE}" ]; then
  echo "Usage: bash script/run_k8s.sh <script.py> <config.yaml>" >&2
  exit 2
fi

if ! command -v kubectl >/dev/null 2>&1; then
  echo "kubectl is required to submit Ray jobs to Kubernetes." >&2
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

NAMESPACE=${K8S_NAMESPACE:-kaiyuan-ray}
SERVICE_ACCOUNT=${K8S_SERVICE_ACCOUNT:-ray-job-runner}
RAY_JOB_IMAGE=${RAY_JOB_IMAGE:-kaiyuan-ray-app:latest}
RAY_JOB_IMAGE_PULL_POLICY=${RAY_JOB_IMAGE_PULL_POLICY:-IfNotPresent}
RAY_ADDRESS=${RAY_ADDRESS:-ray://raycluster-kaiyuan-head-svc:10001}
WAIT_TIMEOUT=${K8S_WAIT_TIMEOUT:-3600s}
DELETE_SUBMIT_JOB=${K8S_DELETE_SUBMIT_JOB:-1}

raw_job_name=${K8S_SUBMIT_JOB_NAME:-ray-pipeline-$(date +%s)-$RANDOM}
JOB_NAME=$(echo "${raw_job_name}" | tr '[:upper:]' '[:lower:]' | tr -c 'a-z0-9-' '-')
JOB_NAME=${JOB_NAME#-}
JOB_NAME=${JOB_NAME%-}
JOB_NAME=${JOB_NAME:0:63}
JOB_NAME=${JOB_NAME%-}

if [ -z "${JOB_NAME}" ]; then
  JOB_NAME="ray-pipeline-$(date +%s)"
fi

echo "Submitting Ray pipeline job '${JOB_NAME}' in namespace '${NAMESPACE}'"

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
        - name: ray-pipeline
          image: ${RAY_JOB_IMAGE}
          imagePullPolicy: ${RAY_JOB_IMAGE_PULL_POLICY}
          workingDir: /workspace
          command: ["/bin/bash", "-lc"]
          args:
            - |
              set -euo pipefail
              python ${script_in_container} --config ${config_in_container} --mode k8s --ray-address ${RAY_ADDRESS}
EOF_YAML

if ! kubectl -n "${NAMESPACE}" wait --for=condition=complete "job/${JOB_NAME}" --timeout="${WAIT_TIMEOUT}"; then
  echo "Ray pipeline job failed or timed out: ${JOB_NAME}" >&2
  kubectl -n "${NAMESPACE}" describe "job/${JOB_NAME}" || true
  kubectl -n "${NAMESPACE}" logs "job/${JOB_NAME}" --all-containers=true || true
  exit 1
fi

kubectl -n "${NAMESPACE}" logs "job/${JOB_NAME}" --all-containers=true

if [ "${DELETE_SUBMIT_JOB}" = "1" ]; then
  kubectl -n "${NAMESPACE}" delete job "${JOB_NAME}" --ignore-not-found >/dev/null 2>&1 || true
fi
