from __future__ import annotations

import argparse
from typing import Tuple

import yaml

from datafiner.base import PipelineNode, PipelineTree
from datafiner.dataset_utils import (
    dataset_from_pandas,
    normalize_lance_path,
    select_columns,
    union_children,
)
from datafiner.register import register


@register("Splitter")
class Splitter(PipelineNode):
    """
    Split a dataset into train and validation sets and write both to Lance.
    """

    def __init__(
        self,
        runtime,
        train_file: str,
        val_file: str,
        num_train: int,
        split_method: str = "exact",
        shuffle: bool = False,
        num_train_files: int = None,
        num_val_files: int = None,
        mode: str = "overwrite",
        select_cols: list = None,
        shuffle_train: bool = False,
        child_configs: list = None,
        storage_options: dict | None = None,
    ) -> None:
        super().__init__(runtime, child_configs)

        if not isinstance(num_train, int) or num_train <= 0:
            raise ValueError("'num_train' must be a positive integer.")
        if split_method not in ["exact", "fast_approximate"]:
            raise ValueError("'split_method' must be either 'exact' or 'fast_approximate'.")

        self.train_file = train_file
        self.val_file = val_file
        self.num_train = num_train
        self.split_method = split_method
        self.shuffle = shuffle
        self.num_train_files = num_train_files
        self.num_val_files = num_val_files
        self.mode = mode
        self.select_cols = select_cols
        self.shuffle_train = shuffle_train
        self.storage_options = storage_options or self.runtime.storage_options

    def run(self):
        ds = union_children(self.children, by_name=False)

        if self.shuffle and self.split_method == "exact":
            print("Shuffle is enabled for 'exact' split. Shuffling the full dataset...")
            ds = ds.random_shuffle(seed=42)

        if self.select_cols is not None:
            ds = select_columns(ds, self.select_cols, runtime=self.runtime)

        if self.split_method == "exact":
            train_ds, val_ds = self._split_exact(ds)
        else:
            train_ds, val_ds = self._split_fast_approximate(ds)

        if self.shuffle_train:
            train_ds = train_ds.random_shuffle(seed=42)

        self._write_split(train_ds, self.train_file, self.num_train_files)
        self._write_split(val_ds, self.val_file, self.num_val_files)

        print("Splitting and writing completed successfully.")
        return val_ds

    def _write_split(self, ds, path: str, num_files: int):
        if num_files:
            ds = ds.repartition(num_files, shuffle=False)

        mode = self.mode
        ray_mode = {
            "overwrite": "overwrite",
            "append": "append",
            "ignore": "create",
            "create": "create",
        }.get(mode, "overwrite")

        print(f"Writing split to {path} (mode={ray_mode})")
        ds.write_lance(
            normalize_lance_path(path),
            mode=ray_mode,
            storage_options=self.storage_options,
        )

    def _split_exact(self, ds) -> Tuple:
        print(f"Using 'exact' split method for {self.num_train} training rows.")
        if hasattr(ds, "split_at_indices"):
            return ds.split_at_indices([self.num_train])

        pdf = ds.to_pandas()
        if pdf.empty:
            empty = dataset_from_pandas(pdf)
            return empty, empty

        train_pdf = pdf.head(self.num_train).reset_index(drop=True)
        val_pdf = pdf.iloc[self.num_train :].reset_index(drop=True)
        return dataset_from_pandas(train_pdf), dataset_from_pandas(val_pdf)

    def _split_fast_approximate(self, ds) -> Tuple:
        print(
            f"Using 'fast_approximate' split method for roughly {self.num_train} training rows."
        )
        total_count = ds.count()
        if total_count == 0:
            empty = dataset_from_pandas(ds.to_pandas())
            return empty, empty

        if self.num_train >= total_count:
            print(
                "Warning: 'num_train' is >= total rows. All data will be in the training set."
            )
            full = ds
            empty = dataset_from_pandas(ds.limit(0).to_pandas())
            return full, empty

        fraction = self.num_train / total_count
        train_ds = ds.random_sample(fraction=fraction, seed=42)

        train_pdf = train_ds.to_pandas().reset_index(drop=True)
        source_pdf = ds.to_pandas().reset_index(drop=True)

        if train_pdf.empty:
            return dataset_from_pandas(train_pdf), dataset_from_pandas(source_pdf)

        marker_col = "__split_marker__"
        source_pdf[marker_col] = range(len(source_pdf))
        train_pdf = train_pdf.copy()
        train_pdf[marker_col] = range(len(train_pdf))

        cutoff = len(train_pdf)
        train_exact = source_pdf.head(cutoff).drop(columns=[marker_col])
        val_exact = source_pdf.iloc[cutoff:].drop(columns=[marker_col])

        print(
            f"Approximation resulted in {len(train_exact)} training rows (target was {self.num_train})."
        )

        return dataset_from_pandas(train_exact), dataset_from_pandas(val_exact)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--mode", type=str, choices=["local", "k8s"], default="local")
    parser.add_argument("--ray-address", type=str, default=None)

    args = parser.parse_args()
    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    pipeline = PipelineTree(config, mode=args.mode, ray_address=args.ray_address)
    ds = pipeline.run()
    ds.show(20)
    print(ds.count())
