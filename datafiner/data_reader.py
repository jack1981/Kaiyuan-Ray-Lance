"""Dataset source nodes for Lance/Parquet/JSON/NPY ingestion.

These readers translate source-specific options into Ray Data read calls and
normalize post-read projection/repartition/debug behaviors.
They serve as pipeline leaf nodes analogous to raw-dataset leaves in the
PCMind-2.1 tree-structured preprocessing framework.
See also `datafiner/data_writer.py` for sink-side controls.
"""

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
    """Normalize a single path or list of paths into a string list.

    Args:
        input_path: Path-like input from config.

    Returns:
        List of string paths.

    Side effects:
        None.

    Assumptions:
        Callers have already validated semantic path correctness.
    """
    if isinstance(input_path, list):
        return [str(p) for p in input_path]
    return [str(input_path)]


class DataReader(PipelineNode, ABC):
    """Abstract reader node that produces datasets directly from storage inputs.

    Inputs/outputs:
        Accepts source paths and optional projection/repartition settings; `run`
        returns a Ray `Dataset`.

    Side effects:
        Subclasses perform filesystem/object-store reads.

    Assumptions:
        Reader nodes are leaves in the pipeline tree and therefore do not accept
        child datasets.
    """

    def __init__(
        self,
        runtime,
        input_path: Union[str, list],
        num_parallel: int = None,
        child_configs: list = None,
        select_cols: list = None,
    ):
        """Initialize common reader configuration.

        Args:
            runtime: Shared runtime config.
            input_path: Source path or list of source paths.
            num_parallel: Optional post-read block count target.
            child_configs: Unsupported for readers; kept for interface parity.
            select_cols: Optional list of columns to project.

        Returns:
            None.

        Side effects:
            None.

        Assumptions:
            `num_parallel` is interpreted as `repartition(..., shuffle=False)`.
        """
        super().__init__(runtime, child_configs)
        self.input_path = input_path
        self.num_parallel = num_parallel
        self.select_cols = select_cols

    @abstractmethod
    def read(self):
        """Read source data and return a Ray dataset.

        Inputs/outputs:
            No runtime inputs; returns a Ray `Dataset`.

        Side effects:
            Performs source-specific I/O operations.

        Assumptions:
            Subclasses enforce their own format and schema expectations.
        """
        pass

    def run(self):
        """Execute the reader as a leaf node.

        Inputs/outputs:
            No inputs; returns dataset from `read()`.

        Side effects:
            Raises on invalid child configuration and performs read I/O.

        Assumptions:
            Reader nodes should never have upstream child nodes.
        """
        if self.children:
            raise ValueError("DataReader does not support child configs")
        return self.read()


@register("LanceReader")
class LanceReader(DataReader):
    """Read Lance or Parquet sources with auto-format detection and union support.

    Inputs/outputs:
        Consumes one or many source paths and returns a single dataset.

    Side effects:
        Performs remote/local reads, optional repartitioning, and debug logging.

    Assumptions:
        Multi-path reads preserve historical behavior: union by name with missing
        columns allowed.
    """

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
        """Configure Lance/Parquet reader options.

        Args:
            runtime: Shared runtime config.
            input_path: Source path or paths.
            num_parallel: Optional repartition block count after read.
            child_configs: Unsupported for readers.
            select_cols: Optional projection list.
            mergeSchema: Legacy compatibility flag (currently unused by Ray read).
            schema: Legacy compatibility schema option (currently passthrough).
            datetimeRebaseModeInRead: Legacy Spark compatibility knob.
            input_format: `lance`, `parquet`, or `auto`.
            storage_options: Optional read storage options override.

        Returns:
            None.

        Side effects:
            None during initialization.

        Assumptions:
            In `auto` mode, extension/prefix heuristics select reader first and
            then fall back from Lance to Parquet on failure.
        """
        super().__init__(runtime, input_path, num_parallel, child_configs, select_cols)
        self.mergeSchema = mergeSchema
        self.datetimeRebaseModeInRead = datetimeRebaseModeInRead
        self.input_format = input_format.lower()
        self.schema = schema
        self.storage_options = storage_options or self.runtime.storage_options

    def _looks_like_lance_path(self, path: str) -> bool:
        """Heuristically detect Lance-formatted paths.

        Inputs/outputs:
            Accepts a path and returns boolean.

        Side effects:
            None.

        Assumptions:
            Historical configs encode Lance with either `lance.` prefix or
            `.lance` directory suffix.
        """
        return path.startswith("lance.") or ".lance" in path

    def _looks_like_parquet_path(self, path: str) -> bool:
        """Heuristically detect Parquet-like paths.

        Inputs/outputs:
            Accepts a path and returns boolean.

        Side effects:
            None.

        Assumptions:
            Existing datasets may use nested `/parquets/` or `_parquet` naming.
        """
        return ".parquet" in path or "/parquets/" in path or "_parquet" in path

    def _read_lance(self, path: str):
        """Read a Lance dataset with optional projection and storage options.

        Inputs/outputs:
            Reads one path and returns a Ray dataset.

        Side effects:
            Performs Lance I/O against local or object-store path.

        Assumptions:
            `normalize_lance_path` has to canonicalize SQL-style path wrappers.
        """
        return ray.data.read_lance(
            normalize_lance_path(path),
            columns=self.select_cols,
            storage_options=self.storage_options,
        )

    def _read_parquet(self, path: str):
        """Read Parquet data and apply optional projection.

        Inputs/outputs:
            Reads one path and returns a Ray dataset.

        Side effects:
            Performs Parquet I/O and optional `map_batches` projection.

        Assumptions:
            Projection is applied manually because `read_parquet` path handling
            differs from Lance's built-in `columns` parameter behavior.
        """
        ds = ray.data.read_parquet(path)
        if self.select_cols is not None:
            ds = select_columns(ds, self.select_cols, runtime=self.runtime)
        return ds

    def _read_one(self, path: str):
        """Read one source path according to explicit or inferred format.

        Inputs/outputs:
            Accepts a single path and returns a dataset.

        Side effects:
            Performs source I/O and may catch/ignore format-detection failures.

        Assumptions:
            In `auto`, failing Lance read implies likely Parquet and is retried.
        """
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
        """Read configured paths and return one combined dataset.

        Inputs/outputs:
            Uses `self.input_path`; returns a dataset possibly unioned from
            multiple source datasets.

        Side effects:
            Performs source reads, optional row count call in debug mode, and
            optional repartitioning.

        Assumptions:
            Multi-input union is by column name with missing columns allowed for
            compatibility with previous Spark behavior.
        """
        paths = _normalize_paths(self.input_path)
        datasets = []
        for path in paths:
            with timed_stage(self.runtime, f"reader.read:{self.__class__.__name__}:{path}"):
                datasets.append(self._read_one(path))

        # NOTE(readability): Union-by-name is kept intentionally so mixed-schema
        # Lance shards remain readable without forcing strict schema equality.
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
    """Read JSON/JSONL(.zst-style path lists) through Ray's JSON reader.

    Inputs/outputs:
        Reads one or many JSON paths and returns a dataset.

    Side effects:
        Performs filesystem/object-store reads and optional repartitioning.

    Assumptions:
        Compression handling is delegated to Ray's JSON datasource.
    """

    def read(self):
        """Read JSONL sources, then apply projection and repartition settings.

        Inputs/outputs:
            No inputs; returns dataset from `self.input_path`.

        Side effects:
            Performs I/O and optional transform/repartition work.

        Assumptions:
            Optional `select_cols` names exist in loaded schema.
        """
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
        """Initialize JSON reader options.

        Args:
            runtime: Shared runtime config.
            input_path: Source path or paths.
            multiLine: Legacy compatibility flag for prior JSON reader behavior.
            num_parallel: Optional post-read repartition block count.
            child_configs: Unsupported for readers.
            select_cols: Optional projection list.

        Returns:
            None.

        Side effects:
            None.

        Assumptions:
            Ray JSON reader behavior is used for both single-line and multiline
            content in this migration.
        """
        super().__init__(runtime, input_path, num_parallel, child_configs, select_cols)
        self.multiLine = multiLine

    def read(self):
        """Read JSON files and apply optional projection/repartition.

        Inputs/outputs:
            Reads paths from `self.input_path`; returns dataset.

        Side effects:
            Prints info line and performs read/transformation I/O.

        Assumptions:
            `multiLine` is informational for compatibility, not passed to Ray.
        """
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
        """Initialize generic format reader configuration.

        Args:
            runtime: Shared runtime config.
            input_path: Source path or paths.
            data_format: One of `lance`, `parquet`, `json/jsonl`, or `csv`.
            num_parallel: Optional post-read repartition block count.
            child_configs: Unsupported for readers.
            select_cols: Optional projection list.
            storage_options: Optional storage options override for Lance reads.

        Returns:
            None.

        Side effects:
            None.

        Assumptions:
            Non-Lance formats rely on Ray native readers without extra options.
        """
        super().__init__(runtime, input_path, num_parallel, child_configs, select_cols)
        self.data_format = data_format.lower()
        self.storage_options = storage_options or self.runtime.storage_options

    def read(self):
        """Read data according to `data_format` and normalize post-read behavior.

        Inputs/outputs:
            No inputs; returns one dataset from single or multi-path sources.

        Side effects:
            Performs source reads and optional projection/repartition.

        Assumptions:
            Multi-path Lance reads are unioned in order to preserve deterministic
            behavior across path lists.
        """
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
    """Read `.npy` files into rows with array payloads and optional file path.

    Inputs/outputs:
        Expands glob patterns, parses NumPy arrays, and returns item-based Ray
        datasets.

    Side effects:
        Reads local files and logs parse warnings.

    Assumptions:
        Arrays are JSON-serializable after `.tolist()` conversion.
    """

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
        """Initialize NPY reader settings.

        Args:
            runtime: Shared runtime config.
            input_path: Path, glob, or list of either.
            column_name: Output column storing array values.
            path_column: Optional output column storing source file path.
            num_parallel: Optional post-read repartition block count.
            child_configs: Unsupported for readers.
            select_cols: Optional projection list.

        Returns:
            None.

        Side effects:
            None.

        Assumptions:
            Input paths are local filesystem paths accessible from the worker.
        """
        super().__init__(runtime, input_path, num_parallel, child_configs, select_cols)
        if not column_name:
            raise ValueError("'column_name' must be a non-empty string.")
        self.column_name = column_name
        self.path_column = path_column

    def read(self):
        """Load matching NPY files and convert each file into one dataset row.

        Inputs/outputs:
            Uses configured input paths and returns a Ray dataset.

        Side effects:
            Performs local filesystem glob/read operations and optional warnings.

        Assumptions:
            Invalid/corrupt files are skipped instead of failing the full read.
        """
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
        """Wrap a dataset with a `run()` method for union helper compatibility.

        Inputs/outputs:
            Stores an in-memory dataset; returns nothing.

        Side effects:
            None.

        Assumptions:
            Used only as a temporary adapter inside reader union code paths.
        """
        self._ds = ds

    def run(self):
        """Return the wrapped dataset.

        Inputs/outputs:
            No inputs; returns wrapped dataset.

        Side effects:
            None.

        Assumptions:
            Dataset object remains valid for downstream union operations.
        """
        return self._ds
