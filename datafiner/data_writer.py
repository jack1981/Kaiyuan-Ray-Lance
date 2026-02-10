from __future__ import annotations

from abc import ABC, abstractmethod

from datafiner.base import PipelineNode
from datafiner.dataset_utils import (
    normalize_lance_path,
    path_exists,
    select_columns,
    union_children,
)
from datafiner.register import register


class DataWriter(PipelineNode, ABC):
    def __init__(
        self,
        runtime,
        output_path: str,
        shuffle: bool = True,
        select_cols: list = None,
        child_configs: list = None,
    ):
        super().__init__(runtime, child_configs)
        self.output_path = output_path
        self.shuffle = shuffle
        self.select_cols = select_cols

    @abstractmethod
    def write(self, ds):
        pass

    def run(self):
        ds = union_children(self.children, by_name=False)
        if self.select_cols:
            ds = select_columns(ds, self.select_cols)
        if self.shuffle:
            ds = ds.random_shuffle()
        return self.write(ds)


@register("LanceWriter")
class LanceWriter(DataWriter):
    def __init__(
        self,
        runtime,
        output_path: str,
        shuffle: bool = False,
        num_output_files: int = None,
        num_read_partitions: int = None,
        mode: str = "overwrite",
        select_cols: list = None,
        child_configs: list = None,
        storage_options: dict | None = None,
    ):
        super().__init__(runtime, output_path, shuffle, select_cols, child_configs)
        self.num_output_files = num_output_files
        self.num_read_partitions = num_read_partitions
        self.mode = mode
        self.storage_options = storage_options or self.runtime.storage_options

    def _read_existing(self):
        ds = None
        try:
            ds = __import__("ray").data.read_lance(
                normalize_lance_path(self.output_path),
                storage_options=self.storage_options,
            )
        except Exception:
            ds = None

        if ds is None:
            return None

        if self.num_read_partitions is not None and self.num_read_partitions > 0:
            ds = ds.repartition(self.num_read_partitions)
        return ds

    def run(self):
        if self.mode == "read_if_exists":
            print(
                f"[LanceWriter] Mode 'read_if_exists' set. Checking cache: {self.output_path}"
            )
            existing = self._read_existing()
            if existing is not None:
                print("[LanceWriter] Cache hit. Reading from path.")
                return existing
            print("[LanceWriter] Cache miss. Proceeding to compute and write.")

        print("[LanceWriter] Computing dataset from children...")
        ds = union_children(self.children, by_name=False)
        if self.select_cols:
            ds = select_columns(ds, self.select_cols)
        if self.shuffle:
            ds = ds.random_shuffle()
        return self.write(ds)

    def write(self, ds):
        if self.num_output_files is not None and self.num_output_files > 0:
            print(
                f"[LanceWriter] Repartitioning dataset to {self.num_output_files} blocks for write."
            )
            ds = ds.repartition(self.num_output_files)

        effective_mode = self.mode
        if self.mode == "read_if_exists":
            effective_mode = "overwrite"

        if effective_mode == "ignore" and path_exists(self.output_path):
            print(f"[LanceWriter] Mode 'ignore': target exists, skipping write {self.output_path}")
            return ds

        ray_mode = {
            "overwrite": "overwrite",
            "append": "append",
            "ignore": "create",
            "create": "create",
        }.get(effective_mode, "create")

        print(f"[LanceWriter] Writing data to {self.output_path} (mode={ray_mode})")
        ds.write_lance(
            normalize_lance_path(self.output_path),
            mode=ray_mode,
            storage_options=self.storage_options,
        )
        return ds


@register("LanceWriterZstd")
class LanceWriterZstd(LanceWriter):
    """
    Lance writer with extra repartition controls.
    """

    def __init__(
        self,
        runtime,
        output_path: str,
        shuffle: bool = False,
        num_output_files: int = None,
        num_read_partitions: int = None,
        mode: str = "overwrite",
        select_cols: list = None,
        child_configs: list = None,
        compression_level: int = 9,
        use_coalesce: bool = False,
        merge_count: int = 128,
        storage_options: dict | None = None,
    ):
        super().__init__(
            runtime,
            output_path,
            shuffle,
            num_output_files,
            num_read_partitions,
            mode,
            select_cols,
            child_configs,
            storage_options,
        )
        self.compression_level = compression_level
        self.use_coalesce = use_coalesce
        self.merge_count = merge_count

    def write(self, ds):
        # Ray Data does not expose coalesce semantics directly, so we model it with repartition.
        if self.use_coalesce and self.num_output_files is not None and self.num_output_files > 0:
            ds = ds.repartition(self.num_output_files, shuffle=False)
        elif self.use_coalesce:
            target = max(1, int(ds.num_blocks() / max(self.merge_count, 1)))
            ds = ds.repartition(target, shuffle=False)

        return super().write(ds)
