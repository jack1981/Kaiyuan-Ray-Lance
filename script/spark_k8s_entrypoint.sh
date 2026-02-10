#!/usr/bin/env bash

set -euo pipefail

if [ "${1:-}" = "driver" ]; then
  shift
  # In Kubernetes cluster mode the driver container must run in client mode.
  # If cluster mode is used here, Spark recursively resubmits itself and
  # collides on the same driver pod name.
  exec /opt/spark/bin/spark-submit \
    --deploy-mode client \
    --conf "spark.driver.bindAddress=${SPARK_DRIVER_BIND_ADDRESS:-0.0.0.0}" \
    --conf "spark.executorEnv.SPARK_DRIVER_POD_IP=${SPARK_DRIVER_BIND_ADDRESS:-0.0.0.0}" \
    "$@"
fi

if [ "${1:-}" = "executor" ]; then
  shift
  : "${SPARK_DRIVER_URL:?SPARK_DRIVER_URL is required for executor mode}"
  : "${SPARK_EXECUTOR_ID:?SPARK_EXECUTOR_ID is required for executor mode}"
  : "${SPARK_EXECUTOR_CORES:?SPARK_EXECUTOR_CORES is required for executor mode}"
  : "${SPARK_APPLICATION_ID:?SPARK_APPLICATION_ID is required for executor mode}"
  : "${SPARK_EXECUTOR_POD_IP:?SPARK_EXECUTOR_POD_IP is required for executor mode}"

  mapfile -t spark_java_opts < <(env | grep '^SPARK_JAVA_OPT_' | sort -t_ -k4 -n | sed 's/[^=]*=//')

  cmd=(
    /opt/spark/bin/spark-class
    "${spark_java_opts[@]}"
    org.apache.spark.scheduler.cluster.k8s.KubernetesExecutorBackend
    --driver-url "${SPARK_DRIVER_URL}"
    --executor-id "${SPARK_EXECUTOR_ID}"
    --bind-address "${SPARK_EXECUTOR_POD_IP}"
    --hostname "${SPARK_EXECUTOR_POD_IP}"
    --cores "${SPARK_EXECUTOR_CORES}"
    --app-id "${SPARK_APPLICATION_ID}"
    --resourceProfileId "${SPARK_RESOURCE_PROFILE_ID:-0}"
  )

  if [ -n "${SPARK_EXECUTOR_POD_NAME:-}" ]; then
    cmd+=(--podName "${SPARK_EXECUTOR_POD_NAME}")
  fi

  if [ "$#" -gt 0 ]; then
    cmd+=("$@")
  fi

  exec "${cmd[@]}"
fi

exec "$@"
