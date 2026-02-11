"""Shared Ray Dataset utilities for runtime tuning, schema handling, and I/O.

This module centralizes cross-node behaviors such as `map_batches` defaults,
debug instrumentation, union alignment, and storage option discovery.
In the PCMind-2.1 context, these helpers back reproducible large-scale
filter/dedup/mix pipelines while keeping resource usage predictable.
See also `datafiner/base.py`, `datafiner/data_reader.py`, and
`datafiner/data_writer.py`.
"""

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
    """Ray Data tuning values applied across readers/transforms/writers.

    Inputs/outputs:
        Stores batch sizing, block sizing, and write block cap parameters.

    Side effects:
        None by itself; consumed by context/config helper functions.

    Assumptions:
        Positive integers represent explicit settings; non-positive values map
        to unset (`None`) in `from_dict`.
    """

    batch_size: int | None = 1024
    target_block_size_mb: int | None = 128
    concurrency: int | None = None
    max_write_blocks: int | None = 64

    @classmethod
    def from_dict(cls, value: dict | None) -> "RuntimeDataConfig":
        """Build a typed runtime-data config from possibly loose YAML values.

        Args:
            value: Optional config mapping from `ray.data`.

        Returns:
            A normalized `RuntimeDataConfig`.

        Side effects:
            None.

        Assumptions:
            Integer-like strings are acceptable; booleans are rejected to avoid
            accidental truthy/falsy coercion.
        """
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
    """Execution-time settings shared by every pipeline node.

    Inputs/outputs:
        Carries app identity, Ray connectivity, storage options, debug mode, and
        nested Ray Data tuning values.

    Side effects:
        None; this is a pure data container.

    Assumptions:
        Instances are treated as immutable configuration after pipeline startup.
    """

    app_name: str
    mode: str
    ray_address: str | None = None
    storage_options: dict | None = None
    debug_stats: bool = False
    data: RuntimeDataConfig = field(default_factory=RuntimeDataConfig)


def _coerce_optional_int(value, default: int | None) -> int | None:
    """Parse optional integer-like config values with consistent null semantics.

    Args:
        value: Raw config value.
        default: Value to use when `value` is missing (`None`).

    Returns:
        Parsed positive integer, `None` for non-positive values, or `default`.

    Side effects:
        None.

    Assumptions:
        Booleans indicate configuration mistakes and should raise.
    """
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
    """Apply cluster-wide Ray Data defaults after `ray.init`.

    Args:
        data_config: Parsed runtime data tuning options.

    Returns:
        None.

    Side effects:
        Mutates the process-global Ray Data `DataContext`.

    Assumptions:
        Called after Ray runtime startup; unsupported context fields are ignored.
    """
    context = ray.data.DataContext.get_current()

    if data_config.target_block_size_mb is not None:
        target_bytes = int(data_config.target_block_size_mb * 1024 * 1024)
        if hasattr(context, "target_max_block_size"):
            context.target_max_block_size = target_bytes


def debug_enabled(runtime: RuntimeConfig | None) -> bool:
    """Return whether Ray debug instrumentation should be emitted.

    Inputs/outputs:
        Accepts optional runtime config and returns a boolean flag.

    Side effects:
        None.

    Assumptions:
        Missing runtime means debug logging must stay disabled.
    """
    return bool(runtime and runtime.debug_stats)


def debug_log(runtime: RuntimeConfig | None, message: str) -> None:
    """Print a debug line when runtime-level debug mode is enabled.

    Args:
        runtime: Optional runtime settings.
        message: Message body without prefix.

    Returns:
        None.

    Side effects:
        Writes to stdout.

    Assumptions:
        Debug logs should be prefixed consistently for grep/filtering.
    """
    if debug_enabled(runtime):
        print(f"[RayDebug] {message}")


@contextmanager
def timed_stage(runtime: RuntimeConfig | None, label: str):
    """Context manager that times a stage and logs elapsed seconds in debug mode.

    Args:
        runtime: Optional runtime settings.
        label: Stable stage label for timing output.

    Yields:
        Control to wrapped code block.

    Side effects:
        Emits stdout debug timing when enabled.

    Assumptions:
        Timing overhead should be near-zero when debug mode is off.
    """
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
    """Emit object store capacity headroom diagnostics for a stage.

    Inputs/outputs:
        Consumes runtime config and stage label; returns nothing.

    Side effects:
        Queries cluster resources and prints warning/debug lines.

    Assumptions:
        Resource keys may be absent on some runtimes; failures are best-effort.
    """
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
    """Return dataset block count with optional fallback materialization.

    Args:
        ds: Dataset to inspect.
        materialize_if_needed: Whether to materialize when lazy plans do not
            expose `num_blocks`.

    Returns:
        Tuple of `(num_blocks_or_none, dataset_reference)` where the returned
        dataset may be the materialized object.

    Side effects:
        May trigger dataset materialization and underlying Ray execution.

    Assumptions:
        Block introspection errors should degrade to `None`, not fail callers.
    """
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
    """Log block counts, operator stats, and object-store headroom for a dataset.

    Args:
        runtime: Optional runtime settings.
        ds: Dataset to inspect.
        stage: Label used in emitted messages.

    Returns:
        None.

    Side effects:
        May trigger `ds.stats()` execution and writes diagnostic output.

    Assumptions:
        Logging must never fail pipeline execution; all inspection errors are
        swallowed and reported as debug lines.
    """
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
    """Run `Dataset.map_batches` with runtime defaults and compatibility guards.

    Args:
        ds: Input dataset.
        runtime: Optional runtime config containing default batch knobs.
        fn: Batch transform callable.
        batch_format: Ray batch format forwarded to `map_batches`.
        batch_size: Optional explicit batch size override.
        concurrency: Optional explicit concurrency override.
        **kwargs: Additional Ray `map_batches` options.

    Returns:
        Transformed dataset.

    Side effects:
        Schedules distributed batch transform tasks in Ray.

    Assumptions:
        Defaults should only be injected when parameters are supported by the
        installed Ray version and absent from explicit kwargs.
    """
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

    # NOTE(readability): We only pass knobs supported by the installed Ray
    # version so config remains forward/backward compatible across environments.
    return ds.map_batches(fn, batch_format=batch_format, **options)


def cap_dataset_blocks_for_write(
    ds: ray.data.Dataset, runtime: RuntimeConfig | None
) -> ray.data.Dataset:
    """Cap dataset block count before writing to reduce tiny output files.

    Args:
        ds: Dataset destined for writer sink.
        runtime: Optional runtime config containing `max_write_blocks`.

    Returns:
        Repartitioned dataset when current block count exceeds cap; otherwise
        original dataset.

    Side effects:
        May materialize and repartition dataset, invoking Ray compute.

    Assumptions:
        This helper is only used when the writer does not already enforce a
        fixed `num_output_files`.
    """
    if runtime is None or runtime.data.max_write_blocks is None:
        return ds

    # NOTE(readability): Materializing here is intentional; lazy datasets may
    # not expose block counts until planned execution is concretized.
    num_blocks, ds = dataset_num_blocks(ds, materialize_if_needed=True)
    if num_blocks is None:
        return ds

    max_blocks = int(runtime.data.max_write_blocks)
    if num_blocks <= max_blocks:
        return ds

    debug_log(runtime, f"Capping write blocks from {num_blocks} to {max_blocks}")
    return ds.repartition(max_blocks, shuffle=False)


def normalize_lance_path(path: str) -> str:
    """Normalize path strings for Ray Lance reader/writer compatibility.

    Args:
        path: Raw configured path which may include `lance.` SQL wrappers.

    Returns:
        A normalized path string, including `s3a://` -> `s3://` translation.

    Side effects:
        None.

    Assumptions:
        Normalization is syntax-only and does not verify remote/local existence.
    """
    normalized = path
    if normalized.startswith("lance.`") and normalized.endswith("`"):
        normalized = normalized[len("lance.`") : -1].replace("``", "`")
    elif normalized.startswith("lance."):
        normalized = normalized[len("lance.") :]
    if normalized.startswith("s3a://"):
        normalized = "s3://" + normalized[len("s3a://") :]
    return normalized


def default_storage_options(explicit: dict | None = None) -> dict | None:
    """Resolve Lance/Ray storage options from explicit config or environment.

    Args:
        explicit: User-provided storage option mapping from YAML.

    Returns:
        Storage options dict or `None` when no credentials/endpoint are set.

    Side effects:
        Reads process environment variables.

    Assumptions:
        MinIO-style endpoints without scheme default to HTTP unless overridden.
    """
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
    """Create a Ray dataset from pandas while preserving empty-frame behavior.

    Args:
        df: Source pandas DataFrame.

    Returns:
        Ray `Dataset` containing `df` rows.

    Side effects:
        Materializes the frame into Ray object store.

    Assumptions:
        Empty datasets should become `from_items([])` for predictable schema-less
        behavior used across existing transforms.
    """
    if df.empty:
        return ray.data.from_items([])
    return ray.data.from_pandas(df.reset_index(drop=True))


def _schema_names(ds: ray.data.Dataset) -> list[str]:
    """Best-effort schema name extraction across Ray schema representations.

    Args:
        ds: Dataset to introspect.

    Returns:
        List of column names, or empty list when unavailable.

    Side effects:
        None beyond schema introspection.

    Assumptions:
        Ray version differences may expose schema names as method, attribute, or
        field objects.
    """
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
    """Rename all columns of a dataset by positional mapping.

    Args:
        ds: Input dataset.
        target_columns: Desired column names in positional order.
        runtime: Optional runtime config for tuned `map_batches`.

    Returns:
        Dataset with renamed columns.

    Side effects:
        Executes a pandas `map_batches` transform.

    Assumptions:
        Input and target column counts must match exactly for positional unions.
    """
    def rename_batch(batch: pd.DataFrame) -> pd.DataFrame:
        """Rename a pandas batch to `target_columns` preserving row order.

        Args:
            batch: Source pandas batch.

        Returns:
            Renamed batch DataFrame.

        Side effects:
            None.

        Assumptions:
            The caller enforces consistent column count across union inputs.
        """
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
    """Align a dataset to a target column order, optionally filling missing cols.

    Args:
        ds: Dataset to align.
        all_columns: Canonical output column order.
        runtime: Optional runtime config.

    Returns:
        Dataset with exactly `all_columns` in order.

    Side effects:
        May run `map_batches` for missing-column fill.

    Assumptions:
        Missing columns are represented as pandas `NA` values.
    """
    existing = _schema_names(ds)
    if existing == all_columns:
        return ds

    existing_set = set(existing)

    def align_batch(batch: pd.DataFrame) -> pd.DataFrame:
        """Pad absent columns in a batch and reorder deterministically.

        Args:
            batch: Source pandas batch.

        Returns:
            Batch with target schema.

        Side effects:
            None.

        Assumptions:
            Added columns should be null-filled to mirror SQL union-by-name rules.
        """
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
    """Union datasets by positional schema compatibility.

    Args:
        datasets: Input datasets in union order.
        runtime: Optional runtime config for rename helper.

    Returns:
        Unioned dataset.

    Side effects:
        Triggers dataset union planning and optional rename transforms.

    Assumptions:
        All datasets must have identical column counts; later datasets may be
        renamed to first-dataset column names.
    """
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
    """Union datasets by column names with optional null-filling for missing cols.

    Args:
        datasets: Input datasets in union order.
        allow_missing_columns: Whether to pad absent columns with nulls.
        runtime: Optional runtime config.

    Returns:
        Unioned dataset aligned by column names.

    Side effects:
        May execute alignment transforms before union.

    Assumptions:
        Without `allow_missing_columns`, all datasets share identical column sets.
    """
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
    """Run child nodes and union their datasets with configured schema strategy.

    Args:
        children: Child node objects exposing `run()`.
        by_name: If True, use union-by-name semantics.
        allow_missing_columns: If True with `by_name`, null-fill missing columns.

    Returns:
        A single unioned dataset from all child outputs.

    Side effects:
        Executes child nodes, which can trigger I/O and Ray compute.

    Assumptions:
        At least one child is present and all children share compatible schemas
        under the selected union mode.
    """
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
    """Project a dataset to the specified columns using batch-wise pandas select.

    Args:
        ds: Source dataset.
        columns: Iterable of columns to keep, in output order.
        runtime: Optional runtime config for tuning.

    Returns:
        Dataset containing only selected columns.

    Side effects:
        Runs a `map_batches` transform.

    Assumptions:
        Requested columns exist in each batch schema.
    """
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
    """Drop columns from a dataset if they are present.

    Args:
        ds: Source dataset.
        columns: Iterable of column names to remove.
        runtime: Optional runtime config for tuning.

    Returns:
        Dataset with specified columns removed when found.

    Side effects:
        Runs a `map_batches` transform.

    Assumptions:
        Missing columns should be ignored to preserve permissive behavior.
    """
    drops = list(columns)
    return map_batches_tuned(
        ds,
        runtime,
        lambda batch: batch.drop(columns=[c for c in drops if c in batch.columns]),
        batch_format="pandas",
    )


def show_dataset(ds: ray.data.Dataset, n: int = 20, vertical: bool = False) -> None:
    """Print a small dataset preview in table or vertical key/value layout.

    Args:
        ds: Dataset to preview.
        n: Number of rows to fetch and print.
        vertical: Whether to print one row per block with key/value pairs.

    Returns:
        None.

    Side effects:
        Executes `take(n)` and prints to stdout.

    Assumptions:
        Preview is for diagnostics only and may trigger upstream computation.
    """
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
    """Check local path existence for writer modes that skip existing outputs.

    Args:
        path: Configured output path.

    Returns:
        True if the normalized path exists locally, else False.

    Side effects:
        Performs local filesystem metadata checks.

    Assumptions:
        Remote object-store URIs are treated as non-checkable and return False.
    """
    normalized = normalize_lance_path(path)
    if normalized.startswith(("s3://", "gs://", "abfs://")):
        return False
    return Path(normalized).exists()
