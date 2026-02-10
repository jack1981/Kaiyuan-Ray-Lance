# Kaiyuan Ray + Lance

[![License](https://img.shields.io/badge/License-Apache-f5de53?&color=f5de53)](LICENSE)
[![arXiv-2512.07612](https://img.shields.io/badge/arXiv-2512.07612-b31b1b.svg?style=flat)](https://arxiv.org/abs/2512.07612)

A tree-structured data preprocessing framework migrated from Spark to Ray Data with Lance I/O.

## Performance Update (What Changed and Why)

- Added Ray Data tuning knobs (`ray.data.*`) so block sizing, batching, and concurrency can be adjusted per environment.
- Added gated debug instrumentation (`ray.debug_stats`) for stage timings and `ds.stats()` without noisy default logs.
- Refactored dataset union behavior to avoid large in-memory pandas materialization in common union paths.
- Added writer-side block capping to reduce small-file output explosion when output partitions are not explicitly set.
- Expanded tests with structural performance guards (block counts, map-batches operators, controlled write partitioning).

## Migration Summary

This branch removes Spark/Yarn runtime paths and replaces them with:

- Ray local mode (`ray.init()` on developer laptop / Docker)
- Ray Kubernetes mode (KubeRay `RayCluster` + `RayJob` submit)
- Ray-native transforms (`map_batches`, `random_shuffle`, dataset-level operations)
- Lance I/O through Ray Data (`read_lance`, `write_lance`)
- Persistent job history UI service for completed `RayJob` records

## Spark to Ray Mapping

| Previous Spark Component | New Ray Component |
| --- | --- |
| `SparkSession` bootstrap | `ray.init()` via `PipelineTree` |
| Spark DataFrame nodes | Ray Data `Dataset` nodes |
| Spark UDF / pandas UDF | `Dataset.map_batches` |
| `spark-submit` local mode | `python main.py --mode local` |
| Spark-on-K8s submit | KubeRay `RayJob` + `--mode k8s --ray-address` |
| Spark Lance catalog | Ray Data `read_lance` / `write_lance` |
| Legacy non-Ray cluster mode | Removed |

## Installation

```bash
pip install -r requirements.txt
```

## CLI

```bash
python main.py --config <config.yaml> --mode local|k8s [--ray-address ray://...] [--debug-stats]
```

- `--mode local`: starts a local Ray runtime if no address is supplied.
- `--mode k8s`: connects to a Ray cluster using `--ray-address`, `ray.address` config, `RAY_ADDRESS`, or default `auto` (recommended for RayJob-in-cluster execution).
- `--debug-stats`: enables stage-level timing and `ds.stats()` logs (same as setting `ray.debug_stats: true` in YAML).

## Local Run (Laptop / Docker)

### 1) Build image

```bash
make build
```

### 2) Prepare local sample Lance datasets + models

```bash
make prepare-sample
```

### 3) Run end-to-end example (read -> transform -> write)

```bash
make run
```

Default example is `example/read_write.yaml`.

### 4) Run all examples

```bash
make run-examples
```

### 5) Manual benchmark (non-CI)

```bash
make bench
```

Useful env vars:

- `PIPELINE_CONFIG=example/read_write.yaml`
- `BENCH_REPEAT=3`
- `BENCH_EXTRA_ARGS=--debug-stats`

## Kubernetes Run (KubeRay)

### 1) Bring up kind + MinIO + KubeRay + RayCluster

```bash
make k8s-up
```

This applies:

- `k8s/base/*` (namespace, RBAC, MinIO)
- KubeRay operator manifest
- `k8s/kuberay/raycluster.yaml`

### 2) Prepare and upload sample assets to MinIO

```bash
make k8s-prepare
```

### 3) Submit pipeline to cluster (as RayJob, retained for history)

```bash
make k8s-run
```

Equivalent direct submit:

```bash
K8S_NAMESPACE=kaiyuan-ray \
RAY_JOB_IMAGE=kaiyuan-ray-app:latest \
RAY_ADDRESS=auto \
bash script/run_k8s.sh main.py example/read_write.yaml
```

### 4) Run all example configs on cluster

```bash
make k8s-run-examples
```

### 5) Open Ray UI + History UI

```bash
make k8s-ui
```

- Ray Dashboard: `http://localhost:30265`
- Ray Job History UI: `http://localhost:30080`

Optional port-forward mode:

```bash
make k8s-dashboard-port-forward
make k8s-history-port-forward
```

### 6) Check historical job status

```bash
make k8s-history
```

`script/run_k8s.sh` submits a `RayJob` CR and keeps it by default, so completed/failed jobs remain visible in history.

### 7) Tear down kind cluster

```bash
make k8s-down
```

## Example Config

`example/read_write.yaml`:

```yaml
ray:
  app_name: read_write
  debug_stats: false
  data:
    batch_size: 1024
    target_block_size_mb: 128
    concurrency: null
    max_write_blocks: 64
pipeline:
  type: LanceWriter
  output_path: data/output/read_write.lance
  child_configs:
    - type: LanceReader
      input_path: "data/sample/pcmind_kaiyuan_2b_sample.lance"
```

## Performance Tuning

### Ray Data knobs

- `ray.debug_stats`: `true` enables timing logs and `ds.stats()` snapshots per stage.
- `ray.data.batch_size`: default `1024`; used by `map_batches` transforms if stage does not override.
- `ray.data.target_block_size_mb`: default `128`; applied to Ray Data context for read/task granularity.
- `ray.data.concurrency`: optional global `map_batches` concurrency cap.
- `ray.data.max_write_blocks`: default `64`; caps write-side blocks when `num_output_files` is not set.

### Recommended defaults

- Laptop/local:
  - `batch_size: 512-2048`
  - `concurrency: null` (let Ray pick) or small explicit value (2-8)
  - `target_block_size_mb: 64-128`
- KubeRay:
  - increase `batch_size` and `concurrency` gradually with memory headroom
  - keep `target_block_size_mb` in `128-512` range for fewer tiny tasks
  - set `num_output_files` for deterministic output layout when needed

### Reading `ds.stats()`

- Check operator count and task fan-out for accidental tiny-block explosions.
- Ensure heavy transforms appear as `MapBatches(...)` operators (not row-wise map operators).
- Confirm repartition/shuffle operators only appear when intentionally configured.

## Regression Testing

- Run `pytest -q` for correctness + structural performance guard tests.
- Structural guard tests focus on:
  - bounded block counts after read/union/transform,
  - controlled write partition counts,
  - map-batches operator presence in dataset stats.

## Lance I/O Notes

- Reader/writer paths support local and object-store URIs (`s3://...`, `s3a://...`).
- `s3a://` is normalized to `s3://` for Lance APIs.
- Runtime object-store options can be provided by:
  - `ray.storage_options` in config
  - `LANCE_STORAGE_OPTIONS_JSON`
  - AWS/MinIO env vars (`MINIO_ENDPOINT`, `MINIO_ACCESS_KEY`, `MINIO_SECRET_KEY`, etc.)

## Compatibility Notes (Spark-Lance -> Ray-Lance)

- Dataset format remains Lance.
- Writer mode mapping:
  - Spark `overwrite` -> Ray `overwrite`
  - Spark `append` -> Ray `append`
  - Spark `ignore` -> Ray `create` if target does not exist
  - Spark `read_if_exists` -> read existing Lance dataset, else compute + overwrite
- Partition control now maps to Ray dataset repartition before Lance writes.

## Troubleshooting

- Ray address connection failures:
  - In-cluster RayJob mode should use `RAY_ADDRESS=auto`.
  - For external clients, verify Ray head service reachability: `ray://raycluster-kaiyuan-head-svc:10001`.
  - Check cluster pods: `kubectl -n kaiyuan-ray get pods`
- Ray Dashboard/UI unavailable:
  - Check services: `kubectl -n kaiyuan-ray get svc ray-dashboard ray-history-server`
  - Use port-forward fallback: `make k8s-dashboard-port-forward` and `make k8s-history-port-forward`
- History table is empty:
  - Verify RayJob resources exist: `kubectl -n kaiyuan-ray get rayjobs`
  - Run a job: `make k8s-run`
- Image pull failures on kind:
  - Re-load local image: `kind load docker-image kaiyuan-ray-app:latest --name kaiyuan-ray`
- Kubernetes permission errors:
  - Ensure `k8s/base/rbac.yaml` is applied and Job uses `ray-job-runner`
- MinIO endpoint issues:
  - For host access in Docker, use `host.docker.internal:30900`
  - For in-cluster access, use `http://minio.kaiyuan-ray.svc.cluster.local:9000`

## Development Layout

- Pipeline runtime: `main.py`, `datafiner/base.py`
- I/O nodes: `datafiner/data_reader.py`, `datafiner/data_writer.py`
- Transform/filter/dedup nodes: `datafiner/*.py`, `datafiner/deduplication/*.py`
- Local/K8s scripts: `script/*.sh`, `script/*.py`
- KubeRay manifests: `k8s/kuberay/*.yaml`
- Example configs: `example/*.yaml`
