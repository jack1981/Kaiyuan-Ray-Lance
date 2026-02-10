from __future__ import annotations

import inspect
import json
import os
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Sequence
from urllib.parse import urlparse

import pandas as pd
import ray
import ray.data


@dataclass
class RuntimeDataConfig:
    """Ray Data tuning defaults tuned for local and small-cluster stability."""

    batch_size: int | None = 1024
    target_block_size_mb: int | None = 128
    concurrency: int | None = None
    max_write_blocks: int | None = 64

    @classmethod
    def from_dict(cls, value: dict | None) -> "RuntimeDataConfig":
        defaults = cls()
        if not isinstance(value, dict):
            return defaults

        return cls(
            batch_size=_coerce_optional_int(value.get("batch_size"), defaults.batch_size),
            target_block_size_mb=_coerce_optional_int(
                value.get("target_block_size_mb"), defaults.target_block_size_mb
            ),
            concurrency=_coerce_optional_int(value.get("concurrency"), defaults.concurrency),
            max_write_blocks=_coerce_optional_int(
                value.get("max_write_blocks"), defaults.max_write_blocks
            ),
        )


@dataclass
class RuntimeConfig:
    app_name: str
    mode: str
    ray_address: str | None = None
    storage_options: dict | None = None
    debug_stats: bool = False
    data: RuntimeDataConfig = field(default_factory=RuntimeDataConfig)


def _coerce_optional_int(value, default: int | None) -> int | None:
    if value is None:
        return default
    if isinstance(value, bool):
        raise ValueError(f"Expected integer, got boolean: {value}")
    parsed = int(value)
    if parsed <= 0:
        return None
    return parsed


try:
    _MAP_BATCHES_PARAMS = set(inspect.signature(ray.data.Dataset.map_batches).parameters)
except Exception:
    _MAP_BATCHES_PARAMS = set()


def configure_data_context(data_config: RuntimeDataConfig) -> None:
    """Apply global Ray Data context defaults once Ray is initialized."""
    context = ray.data.DataContext.get_current()

    if data_config.target_block_size_mb is not None:
        target_bytes = int(data_config.target_block_size_mb * 1024 * 1024)
        if hasattr(context, "target_max_block_size"):
            context.target_max_block_size = target_bytes


def debug_enabled(runtime: RuntimeConfig | None) -> bool:
    return bool(runtime and runtime.debug_stats)


def debug_log(runtime: RuntimeConfig | None, message: str) -> None:
    if debug_enabled(runtime):
        print(f"[RayDebug] {message}")


@contextmanager
def timed_stage(runtime: RuntimeConfig | None, label: str):
    if not debug_enabled(runtime):
        yield
        return

    start = time.perf_counter()
    try:
        yield
    finally:
        elapsed = time.perf_counter() - start
        debug_log(runtime, f"{label} took {elapsed:.3f}s")


def log_object_store_headroom(runtime: RuntimeConfig | None, stage: str) -> None:
    if not debug_enabled(runtime):
        return

    try:
        total = float(ray.cluster_resources().get("object_store_memory", 0.0))
        available = float(ray.available_resources().get("object_store_memory", 0.0))
    except Exception:
        return

    if total <= 0:
        return

    ratio = available / total
    debug_log(
        runtime,
        f"{stage} object-store headroom={ratio:.1%} "
        f"(available={int(available)}, total={int(total)})",
    )
    if ratio < 0.15:
        print(
            f"[RayDebug][Warning] Low object-store headroom after '{stage}' "
            f"({ratio:.1%}). Consider lower concurrency or larger object store."
        )


def dataset_num_blocks(
    ds: ray.data.Dataset, *, materialize_if_needed: bool = False
) -> tuple[int | None, ray.data.Dataset]:
    """Best-effort block count helper for lazy and materialized datasets."""
    try:
        return ds.num_blocks(), ds
    except NotImplementedError:
        if not materialize_if_needed:
            return None, ds
        try:
            materialized = ds.materialize()
            return materialized.num_blocks(), materialized
        except Exception:
            return None, ds
    except Exception:
        return None, ds


def log_dataset_stats(runtime: RuntimeConfig | None, ds: ray.data.Dataset, stage: str) -> None:
    if not debug_enabled(runtime):
        return

    num_blocks, _ = dataset_num_blocks(ds, materialize_if_needed=False)
    if num_blocks is None:
        num_blocks = "unknown"
    debug_log(runtime, f"{stage} num_blocks={num_blocks}")

    try:
        stats = ds.stats()
    except Exception as exc:
        debug_log(runtime, f"{stage} stats unavailable: {exc}")
        stats = None
    if stats:
        print(f"[RayDebug] {stage} stats\n{stats}")

    log_object_store_headroom(runtime, stage)


def map_batches_tuned(
    ds: ray.data.Dataset,
    runtime: RuntimeConfig | None,
    fn,
    *,
    batch_format: str = "pandas",
    batch_size: int | None = None,
    concurrency: int | None = None,
    **kwargs,
) -> ray.data.Dataset:
    """Apply map_batches with runtime defaults and optional per-stage overrides."""
    options = dict(kwargs)

    effective_batch_size = batch_size
    if effective_batch_size is None and runtime is not None:
        effective_batch_size = runtime.data.batch_size
    if (
        effective_batch_size is not None
        and "batch_size" in _MAP_BATCHES_PARAMS
        and "batch_size" not in options
    ):
        options["batch_size"] = int(effective_batch_size)

    effective_concurrency = concurrency
    if effective_concurrency is None and runtime is not None:
        effective_concurrency = runtime.data.concurrency
    if (
        effective_concurrency is not None
        and "concurrency" in _MAP_BATCHES_PARAMS
        and "concurrency" not in options
    ):
        options["concurrency"] = int(effective_concurrency)

    return ds.map_batches(fn, batch_format=batch_format, **options)


def cap_dataset_blocks_for_write(
    ds: ray.data.Dataset, runtime: RuntimeConfig | None
) -> ray.data.Dataset:
    """Prevent small-file explosion when writer partition count is not explicitly set."""
    if runtime is None or runtime.data.max_write_blocks is None:
        return ds

    num_blocks, ds = dataset_num_blocks(ds, materialize_if_needed=True)
    if num_blocks is None:
        return ds

    max_blocks = int(runtime.data.max_write_blocks)
    if num_blocks <= max_blocks:
        return ds

    debug_log(runtime, f"Capping write blocks from {num_blocks} to {max_blocks}")
    return ds.repartition(max_blocks, shuffle=False)


def normalize_lance_path(path: str) -> str:
    normalized = path
    if normalized.startswith("lance.`") and normalized.endswith("`"):
        normalized = normalized[len("lance.`") : -1].replace("``", "`")
    elif normalized.startswith("lance."):
        normalized = normalized[len("lance.") :]
    if normalized.startswith("s3a://"):
        normalized = "s3://" + normalized[len("s3a://") :]
    return normalized


def default_storage_options(explicit: dict | None = None) -> dict | None:
    if explicit:
        return explicit

    options_json = os.getenv("LANCE_STORAGE_OPTIONS_JSON")
    if options_json:
        try:
            parsed = json.loads(options_json)
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            pass

    endpoint = os.getenv("AWS_ENDPOINT_URL") or os.getenv("MINIO_ENDPOINT")
    access_key = os.getenv("AWS_ACCESS_KEY_ID") or os.getenv("MINIO_ACCESS_KEY")
    secret_key = os.getenv("AWS_SECRET_ACCESS_KEY") or os.getenv("MINIO_SECRET_KEY")
    region = os.getenv("AWS_REGION") or os.getenv("MINIO_REGION")
    allow_http = os.getenv("LANCE_AWS_ALLOW_HTTP")
    allow_http_flag: bool | None = None
    if allow_http is not None:
        allow_http_flag = str(allow_http).lower() in {"1", "true", "yes"}

    if endpoint and "://" in endpoint:
        parsed = urlparse(endpoint)
        if allow_http_flag is None and parsed.scheme:
            allow_http_flag = parsed.scheme.lower() == "http"
    elif endpoint:
        if allow_http_flag is None:
            # MinIO-style endpoints are commonly provided without a scheme.
            allow_http_flag = True
        endpoint = f"{'http' if allow_http_flag else 'https'}://{endpoint}"

    if not any([endpoint, access_key, secret_key, region, allow_http_flag is not None]):
        return None

    options = {}
    if endpoint:
        options["aws_endpoint"] = endpoint
    if access_key:
        options["aws_access_key_id"] = access_key
    if secret_key:
        options["aws_secret_access_key"] = secret_key
    if region:
        options["aws_region"] = region
    if allow_http_flag is not None:
        options["aws_allow_http"] = "true" if allow_http_flag else "false"
    return options


def dataset_from_pandas(df: pd.DataFrame) -> ray.data.Dataset:
    if df.empty:
        return ray.data.from_items([])
    return ray.data.from_pandas(df.reset_index(drop=True))


def _schema_names(ds: ray.data.Dataset) -> list[str]:
    try:
        schema = ds.schema()
    except Exception:
        return []

    if schema is None:
        return []

    names_attr = getattr(schema, "names", None)
    if callable(names_attr):
        try:
            return list(names_attr())
        except Exception:
            pass
    if names_attr is not None:
        try:
            return list(names_attr)
        except Exception:
            pass

    try:
        return [field.name for field in schema]
    except Exception:
        return []


def _rename_columns_to(
    ds: ray.data.Dataset, target_columns: list[str], runtime: RuntimeConfig | None
) -> ray.data.Dataset:
    def rename_batch(batch: pd.DataFrame) -> pd.DataFrame:
        out = batch.copy()
        if len(out.columns) != len(target_columns):
            raise ValueError(
                "UnionByPosition requires all datasets to have the same column count."
            )
        out.columns = target_columns
        return out

    return map_batches_tuned(ds, runtime, rename_batch, batch_format="pandas")


def _align_columns_by_name(
    ds: ray.data.Dataset, all_columns: list[str], runtime: RuntimeConfig | None
) -> ray.data.Dataset:
    existing = _schema_names(ds)
    if existing == all_columns:
        return ds

    existing_set = set(existing)

    def align_batch(batch: pd.DataFrame) -> pd.DataFrame:
        out = batch.copy()
        for column in all_columns:
            if column not in out.columns:
                out[column] = pd.NA
        return out[all_columns]

    if set(all_columns).issubset(existing_set):
        return select_columns(ds, all_columns, runtime=runtime)
    return map_batches_tuned(ds, runtime, align_batch, batch_format="pandas")


def _concat_by_position(
    datasets: Sequence[ray.data.Dataset], runtime: RuntimeConfig | None = None
) -> ray.data.Dataset:
    if not datasets:
        raise ValueError("At least one dataset is required.")
    if len(datasets) == 1:
        return datasets[0]

    base_cols = _schema_names(datasets[0])
    result = datasets[0]
    for ds in datasets[1:]:
        current = ds
        current_cols = _schema_names(current)
        if base_cols and current_cols:
            if len(current_cols) != len(base_cols):
                raise ValueError(
                    "UnionByPosition requires all datasets to have the same column count."
                )
            if current_cols != base_cols:
                current = _rename_columns_to(current, base_cols, runtime=runtime)
        result = result.union(current)
    return result


def _concat_by_name(
    datasets: Sequence[ray.data.Dataset],
    allow_missing_columns: bool = False,
    runtime: RuntimeConfig | None = None,
) -> ray.data.Dataset:
    if not datasets:
        raise ValueError("At least one dataset is required.")
    if len(datasets) == 1:
        return datasets[0]

    schema_names = [_schema_names(ds) for ds in datasets]

    if allow_missing_columns:
        all_cols: list[str] = []
        for cols in schema_names:
            for col in cols:
                if col not in all_cols:
                    all_cols.append(col)
        aligned = [
            _align_columns_by_name(ds, all_cols, runtime=runtime) for ds in datasets
        ]
        result = aligned[0]
        for ds in aligned[1:]:
            result = result.union(ds)
        return result

    base_cols = schema_names[0]
    for cols in schema_names[1:]:
        if set(cols) != set(base_cols):
            raise ValueError(
                "UnionByName without allow_missing_columns requires identical columns."
            )

    aligned = [select_columns(ds, base_cols, runtime=runtime) for ds in datasets]
    result = aligned[0]
    for ds in aligned[1:]:
        result = result.union(ds)
    return result


def union_children(
    children: Sequence, by_name: bool = False, allow_missing_columns: bool = False
) -> ray.data.Dataset:
    if not children:
        raise ValueError("Node requires at least one child.")

    runtime = getattr(children[0], "runtime", None)
    datasets = [child.run() for child in children]
    if by_name:
        return _concat_by_name(
            datasets,
            allow_missing_columns=allow_missing_columns,
            runtime=runtime,
        )
    return _concat_by_position(datasets, runtime=runtime)


def select_columns(
    ds: ray.data.Dataset, columns: Iterable[str], runtime: RuntimeConfig | None = None
) -> ray.data.Dataset:
    selected = list(columns)
    return map_batches_tuned(
        ds,
        runtime,
        lambda batch: batch[selected],
        batch_format="pandas",
    )


def drop_columns(
    ds: ray.data.Dataset, columns: Iterable[str], runtime: RuntimeConfig | None = None
) -> ray.data.Dataset:
    drops = list(columns)
    return map_batches_tuned(
        ds,
        runtime,
        lambda batch: batch.drop(columns=[c for c in drops if c in batch.columns]),
        batch_format="pandas",
    )


def show_dataset(ds: ray.data.Dataset, n: int = 20, vertical: bool = False) -> None:
    rows = ds.take(n)
    if not rows:
        print("[]")
        return

    if vertical:
        for idx, row in enumerate(rows):
            print(f"- row {idx}")
            for key, value in row.items():
                print(f"  {key}: {value}")
    else:
        frame = pd.DataFrame(rows)
        print(frame.to_string(index=False))


def path_exists(path: str) -> bool:
    normalized = normalize_lance_path(path)
    if normalized.startswith(("s3://", "gs://", "abfs://")):
        return False
    return Path(normalized).exists()
