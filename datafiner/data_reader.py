from __future__ import annotations

import glob
import io
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Union

import numpy as np
import ray.data

from datafiner.base import PipelineNode
from datafiner.dataset_utils import (
    debug_log,
    log_dataset_stats,
    normalize_lance_path,
    select_columns,
    timed_stage,
    union_children,
)
from datafiner.register import register


def _normalize_paths(input_path: Union[str, list]) -> list[str]:
    if isinstance(input_path, list):
        return [str(p) for p in input_path]
    return [str(input_path)]


class DataReader(PipelineNode, ABC):
    def __init__(
        self,
        runtime,
        input_path: Union[str, list],
        num_parallel: int = None,
        child_configs: list = None,
        select_cols: list = None,
    ):
        super().__init__(runtime, child_configs)
        self.input_path = input_path
        self.num_parallel = num_parallel
        self.select_cols = select_cols

    @abstractmethod
    def read(self):
        pass

    def run(self):
        if self.children:
            raise ValueError("DataReader does not support child configs")
        return self.read()


@register("LanceReader")
class LanceReader(DataReader):
    def __init__(
        self,
        runtime,
        input_path: Union[str, list],
        num_parallel: int = None,
        child_configs: list = None,
        select_cols: list = None,
        mergeSchema: str = "true",
        schema=None,
        datetimeRebaseModeInRead: str = "CORRECTED",
        input_format: str = "auto",
        storage_options: dict | None = None,
    ):
        super().__init__(runtime, input_path, num_parallel, child_configs, select_cols)
        self.mergeSchema = mergeSchema
        self.datetimeRebaseModeInRead = datetimeRebaseModeInRead
        self.input_format = input_format.lower()
        self.schema = schema
        self.storage_options = storage_options or self.runtime.storage_options

    def _looks_like_lance_path(self, path: str) -> bool:
        return path.startswith("lance.") or ".lance" in path

    def _looks_like_parquet_path(self, path: str) -> bool:
        return ".parquet" in path or "/parquets/" in path or "_parquet" in path

    def _read_lance(self, path: str):
        return ray.data.read_lance(
            normalize_lance_path(path),
            columns=self.select_cols,
            storage_options=self.storage_options,
        )

    def _read_parquet(self, path: str):
        ds = ray.data.read_parquet(path)
        if self.select_cols is not None:
            ds = select_columns(ds, self.select_cols, runtime=self.runtime)
        return ds

    def _read_one(self, path: str):
        if self.input_format == "lance":
            return self._read_lance(path)
        if self.input_format == "parquet":
            return self._read_parquet(path)

        if self._looks_like_lance_path(path):
            return self._read_lance(path)
        if self._looks_like_parquet_path(path):
            return self._read_parquet(path)

        try:
            return self._read_lance(path)
        except Exception:
            return self._read_parquet(path)

    def read(self):
        paths = _normalize_paths(self.input_path)
        datasets = []
        for path in paths:
            with timed_stage(self.runtime, f"reader.read:{self.__class__.__name__}:{path}"):
                datasets.append(self._read_one(path))

        # LanceReader unions by name with missing columns allowed, matching prior behavior.
        ds = datasets[0]
        if len(datasets) > 1:
            ds = union_children(
                [
                    _InlineDatasetNode(ds),
                    *[_InlineDatasetNode(item) for item in datasets[1:]],
                ],
                by_name=True,
                allow_missing_columns=True,
            )

        if self.select_cols is not None and self.input_format != "lance":
            ds = select_columns(ds, self.select_cols, runtime=self.runtime)

        if self.runtime.debug_stats:
            with timed_stage(self.runtime, f"reader.count:{self.__class__.__name__}"):
                row_count = ds.count()
            debug_log(self.runtime, f"{self.__class__.__name__} row_count={row_count}")

        if self.num_parallel is not None and self.num_parallel > 0:
            ds = ds.repartition(self.num_parallel, shuffle=False)

        log_dataset_stats(self.runtime, ds, f"reader.output:{self.__class__.__name__}")

        return ds


@register("LanceReaderZstd")
class LanceReaderZstd(LanceReader):
    """
    Lance reader with the same semantics as LanceReader.
    """


@register("JsonlZstReader")
class JsonlZstReader(DataReader):
    def read(self):
        paths = _normalize_paths(self.input_path)
        ds = ray.data.read_json(paths)
        if self.select_cols is not None:
            ds = select_columns(ds, self.select_cols, runtime=self.runtime)
        if self.num_parallel is not None and self.num_parallel > 0:
            ds = ds.repartition(self.num_parallel, shuffle=False)
        log_dataset_stats(self.runtime, ds, f"reader.output:{self.__class__.__name__}")
        return ds


@register("JsonReader")
class JsonReader(DataReader):
    """
    Reads JSON files into Ray datasets.
    """

    def __init__(
        self,
        runtime,
        input_path: Union[str, list],
        multiLine: bool = False,
        num_parallel: int = None,
        child_configs: list = None,
        select_cols: list = None,
    ):
        super().__init__(runtime, input_path, num_parallel, child_configs, select_cols)
        self.multiLine = multiLine

    def read(self):
        print(
            f"INFO: Reading JSON from path: {self.input_path} with multiLine={self.multiLine}"
        )
        paths = _normalize_paths(self.input_path)
        ds = ray.data.read_json(paths)
        if self.select_cols is not None:
            ds = select_columns(ds, self.select_cols, runtime=self.runtime)
        if self.num_parallel is not None and self.num_parallel > 0:
            ds = ds.repartition(self.num_parallel, shuffle=False)
        log_dataset_stats(self.runtime, ds, f"reader.output:{self.__class__.__name__}")
        return ds


@register("FormatReader")
class FormatReader(DataReader):
    """
    Generic format reader backed by Ray Data sources.
    """

    def __init__(
        self,
        runtime,
        input_path: Union[str, list],
        data_format: str = "lance",
        num_parallel: int = None,
        child_configs: list = None,
        select_cols: list = None,
        storage_options: dict | None = None,
    ):
        super().__init__(runtime, input_path, num_parallel, child_configs, select_cols)
        self.data_format = data_format.lower()
        self.storage_options = storage_options or self.runtime.storage_options

    def read(self):
        paths = _normalize_paths(self.input_path)

        if self.data_format == "lance":
            datasets = [
                ray.data.read_lance(
                    normalize_lance_path(path),
                    columns=self.select_cols,
                    storage_options=self.storage_options,
                )
                for path in paths
            ]
            ds = datasets[0]
            for item in datasets[1:]:
                ds = ds.union(item)
            return ds

        if self.data_format == "parquet":
            ds = ray.data.read_parquet(paths)
        elif self.data_format in {"json", "jsonl"}:
            ds = ray.data.read_json(paths)
        elif self.data_format == "csv":
            ds = ray.data.read_csv(paths)
        else:
            raise ValueError(f"Unsupported format for FormatReader: {self.data_format}")

        if self.select_cols is not None:
            ds = select_columns(ds, self.select_cols, runtime=self.runtime)
        if self.num_parallel is not None and self.num_parallel > 0:
            ds = ds.repartition(self.num_parallel, shuffle=False)
        log_dataset_stats(self.runtime, ds, f"reader.output:{self.__class__.__name__}")
        return ds


@register("NpyReader")
class NpyReader(DataReader):
    def __init__(
        self,
        runtime,
        input_path: Union[str, list],
        column_name: str,
        path_column: str = None,
        num_parallel: int = None,
        child_configs: list = None,
        select_cols: list = None,
    ):
        super().__init__(runtime, input_path, num_parallel, child_configs, select_cols)
        if not column_name:
            raise ValueError("'column_name' must be a non-empty string.")
        self.column_name = column_name
        self.path_column = path_column

    def read(self):
        paths = []
        for raw_path in _normalize_paths(self.input_path):
            expanded = glob.glob(raw_path)
            if expanded:
                paths.extend(expanded)
            else:
                if Path(raw_path).exists():
                    paths.append(raw_path)

        rows = []
        for path in sorted(set(paths)):
            try:
                with open(path, "rb") as f:
                    arr = np.load(io.BytesIO(f.read()))
                row = {self.column_name: arr.tolist()}
                if self.path_column:
                    row[self.path_column] = path
                rows.append(row)
            except Exception as exc:
                print(f"Warning: Could not parse npy file '{path}': {exc}")

        if not rows:
            ds = ray.data.from_items([])
        else:
            ds = ray.data.from_items(rows)

        if self.select_cols is not None:
            ds = select_columns(ds, self.select_cols, runtime=self.runtime)

        if self.num_parallel is not None and self.num_parallel > 0:
            ds = ds.repartition(self.num_parallel, shuffle=False)

        log_dataset_stats(self.runtime, ds, f"reader.output:{self.__class__.__name__}")

        return ds


class _InlineDatasetNode:
    """Adapter for reusing union_children with in-memory datasets."""

    def __init__(self, ds):
        self._ds = ds

    def run(self):
        return self._ds
