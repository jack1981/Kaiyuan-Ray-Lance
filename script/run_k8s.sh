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
RAY_CLUSTER_SELECTOR_KEY=${RAY_CLUSTER_SELECTOR_KEY:-ray.io/cluster}
RAY_CLUSTER_SELECTOR_VALUE=${RAY_CLUSTER_SELECTOR_VALUE:-raycluster-kaiyuan}
RAYJOB_SUBMISSION_MODE=${RAYJOB_SUBMISSION_MODE:-K8sJobMode}
RAYJOB_DELETION_POLICY=${RAYJOB_DELETION_POLICY:-DeleteNone}
RAYJOB_TTL_SECONDS=${RAYJOB_TTL_SECONDS:-0}
RAYJOB_SHUTDOWN_AFTER_FINISH=${RAYJOB_SHUTDOWN_AFTER_FINISH:-false}
WAIT_TIMEOUT_SECONDS=${K8S_WAIT_TIMEOUT_SECONDS:-3600}
POLL_INTERVAL_SECONDS=${K8S_POLL_INTERVAL_SECONDS:-5}
DELETE_RAYJOB=${K8S_DELETE_RAYJOB:-0}

raw_job_name=${K8S_RAYJOB_NAME:-rayjob-$(date +%s)-$RANDOM}
JOB_NAME=$(echo "${raw_job_name}" | tr '[:upper:]' '[:lower:]' | tr -c 'a-z0-9-' '-')
JOB_NAME=${JOB_NAME#-}
JOB_NAME=${JOB_NAME%-}
JOB_NAME=${JOB_NAME:0:63}
JOB_NAME=${JOB_NAME%-}

if [ -z "${JOB_NAME}" ]; then
  JOB_NAME="rayjob-$(date +%s)"
fi

echo "Submitting RayJob '${JOB_NAME}' in namespace '${NAMESPACE}'"

kubectl -n "${NAMESPACE}" delete rayjob "${JOB_NAME}" --ignore-not-found >/dev/null 2>&1 || true

cat <<EOF_YAML | kubectl apply -f -
apiVersion: ray.io/v1
kind: RayJob
metadata:
  name: ${JOB_NAME}
  namespace: ${NAMESPACE}
  labels:
    app.kubernetes.io/name: kaiyuan-ray-pipeline
spec:
  entrypoint: |
    python ${script_in_container} --config ${config_in_container} --mode k8s --ray-address ${RAY_ADDRESS}
  submissionMode: ${RAYJOB_SUBMISSION_MODE}
  clusterSelector:
    ${RAY_CLUSTER_SELECTOR_KEY}: ${RAY_CLUSTER_SELECTOR_VALUE}
  shutdownAfterJobFinishes: ${RAYJOB_SHUTDOWN_AFTER_FINISH}
  deletionPolicy: ${RAYJOB_DELETION_POLICY}
  ttlSecondsAfterFinished: ${RAYJOB_TTL_SECONDS}
  submitterPodTemplate:
    spec:
      serviceAccountName: ${SERVICE_ACCOUNT}
      containers:
        - name: submitter
          image: ${RAY_JOB_IMAGE}
          imagePullPolicy: ${RAY_JOB_IMAGE_PULL_POLICY}
EOF_YAML

start_ts=$(date +%s)
last_status=""

while true; do
  job_status=$(kubectl -n "${NAMESPACE}" get rayjob "${JOB_NAME}" -o jsonpath='{.status.jobStatus}' 2>/dev/null || true)
  deploy_status=$(kubectl -n "${NAMESPACE}" get rayjob "${JOB_NAME}" -o jsonpath='{.status.jobDeploymentStatus}' 2>/dev/null || true)
  message=$(kubectl -n "${NAMESPACE}" get rayjob "${JOB_NAME}" -o jsonpath='{.status.message}' 2>/dev/null || true)

  state_line="jobStatus=${job_status:-<none>} deploymentStatus=${deploy_status:-<none>}"
  if [ "${state_line}" != "${last_status}" ]; then
    echo "RayJob state: ${state_line}"
    if [ -n "${message}" ]; then
      echo "RayJob message: ${message}"
    fi
    last_status="${state_line}"
  fi

  job_upper=$(echo "${job_status}" | tr '[:lower:]' '[:upper:]')
  deploy_upper=$(echo "${deploy_status}" | tr '[:lower:]' '[:upper:]')

  if [[ "${job_upper}" == "SUCCEEDED" || "${job_upper}" == "SUCCESS" || "${deploy_upper}" == "COMPLETE" ]]; then
    break
  fi

  if [[ "${job_upper}" == "FAILED" || "${job_upper}" == "STOPPED" || "${deploy_upper}" == "FAILED" ]]; then
    echo "RayJob failed: ${JOB_NAME}" >&2
    kubectl -n "${NAMESPACE}" describe rayjob "${JOB_NAME}" || true
    exit 1
  fi

  now_ts=$(date +%s)
  if [ $((now_ts - start_ts)) -ge "${WAIT_TIMEOUT_SECONDS}" ]; then
    echo "Timed out waiting for RayJob: ${JOB_NAME}" >&2
    kubectl -n "${NAMESPACE}" describe rayjob "${JOB_NAME}" || true
    exit 1
  fi

  sleep "${POLL_INTERVAL_SECONDS}"
done

kubectl -n "${NAMESPACE}" get rayjob "${JOB_NAME}" -o wide

echo "RayJob completed and is retained for history."

echo "History: kubectl -n ${NAMESPACE} get rayjobs"

if [ "${DELETE_RAYJOB}" = "1" ]; then
  kubectl -n "${NAMESPACE}" delete rayjob "${JOB_NAME}" --ignore-not-found >/dev/null 2>&1 || true
fi
