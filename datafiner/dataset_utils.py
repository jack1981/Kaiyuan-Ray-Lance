from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import pandas as pd
import ray
import ray.data


@dataclass
class RuntimeConfig:
    app_name: str
    mode: str
    ray_address: str | None = None
    storage_options: dict | None = None


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

    if not any([endpoint, access_key, secret_key, region, allow_http]):
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
    if allow_http is not None:
        options["aws_allow_http"] = allow_http.lower() in {"1", "true", "yes"}
    return options


def dataset_from_pandas(df: pd.DataFrame) -> ray.data.Dataset:
    if df.empty:
        return ray.data.from_items([])
    return ray.data.from_pandas(df.reset_index(drop=True))


def _concat_by_position(datasets: Sequence[ray.data.Dataset]) -> ray.data.Dataset:
    if not datasets:
        raise ValueError("At least one dataset is required.")
    if len(datasets) == 1:
        return datasets[0]

    frames = [ds.to_pandas() for ds in datasets]
    base_cols = list(frames[0].columns)
    aligned = []
    for frame in frames:
        if len(frame.columns) != len(base_cols):
            raise ValueError(
                "UnionByPosition requires all datasets to have the same column count."
            )
        adjusted = frame.copy()
        adjusted.columns = base_cols
        aligned.append(adjusted)
    return dataset_from_pandas(pd.concat(aligned, ignore_index=True))


def _concat_by_name(
    datasets: Sequence[ray.data.Dataset], allow_missing_columns: bool = False
) -> ray.data.Dataset:
    if not datasets:
        raise ValueError("At least one dataset is required.")
    if len(datasets) == 1:
        return datasets[0]

    frames = [ds.to_pandas() for ds in datasets]
    if allow_missing_columns:
        all_cols = []
        for frame in frames:
            for col in frame.columns:
                if col not in all_cols:
                    all_cols.append(col)
        aligned = []
        for frame in frames:
            expanded = frame.copy()
            for col in all_cols:
                if col not in expanded.columns:
                    expanded[col] = pd.NA
            aligned.append(expanded[all_cols])
        return dataset_from_pandas(pd.concat(aligned, ignore_index=True))

    base_cols = list(frames[0].columns)
    for frame in frames[1:]:
        if set(frame.columns) != set(base_cols):
            raise ValueError(
                "UnionByName without allow_missing_columns requires identical columns."
            )
    aligned = [frame[base_cols] for frame in frames]
    return dataset_from_pandas(pd.concat(aligned, ignore_index=True))


def union_children(
    children: Sequence, by_name: bool = False, allow_missing_columns: bool = False
) -> ray.data.Dataset:
    if not children:
        raise ValueError("Node requires at least one child.")
    datasets = [child.run() for child in children]
    if by_name:
        return _concat_by_name(datasets, allow_missing_columns=allow_missing_columns)
    return _concat_by_position(datasets)


def select_columns(ds: ray.data.Dataset, columns: Iterable[str]) -> ray.data.Dataset:
    selected = list(columns)
    return ds.map_batches(
        lambda batch: batch[selected],
        batch_format="pandas",
    )


def drop_columns(ds: ray.data.Dataset, columns: Iterable[str]) -> ray.data.Dataset:
    drops = list(columns)
    return ds.map_batches(
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
