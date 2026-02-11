"""Sampling and expansion nodes for duplicate-count driven workflows.

These operators either adjust sample multipliers, stochastically round them, or
materialize expanded rows from count/list-style columns.
They are used to express strategic selective repetition behavior similar to the
report's phase-wise high-quality sample repetition policy.
"""

import numpy as np
import pandas as pd

from datafiner.base import PipelineNode
from datafiner.dataset_utils import dataset_from_pandas, map_batches_tuned, union_children
from datafiner.register import register


@register("DuplicateSampleRatio")
class DuplicateSampleRatio(PipelineNode):
    """Scale and cap duplicate-count sampling weights.

    Inputs/outputs:
        Reads child dataset(s) and returns dataset with adjusted count column.

    Side effects:
        Executes pandas batch transforms.

    Assumptions:
        Count column is numeric-coercible and negative values are not expected.
        This is typically used to encode phase-specific selective repetition.
    """

    def __init__(
        self,
        runtime,
        child_configs: list = None,
        global_sample_rate: float = 0.1,
        max_sample: float = 20.0,
        col: str = "duplicate_count",
    ):
        """Configure duplicate sampling ratio transform.

        Args:
            runtime: Shared runtime config.
            child_configs: Upstream node configs.
            global_sample_rate: Multiplier applied to count column.
            max_sample: Upper bound after scaling.
            col: Count column name.

        Returns:
            None.

        Side effects:
            None.

        Assumptions:
            Input column exists and can be coerced to numeric values.
        """
        super().__init__(runtime, child_configs)
        self.global_sample_rate = global_sample_rate
        self.max_sample = max_sample
        self.col = col

    def run(self):
        """Scale duplicate-count column by configured ratio and cap.

        Inputs/outputs:
            Reads child dataset(s) and returns transformed dataset.

        Side effects:
            Runs a pandas `map_batches` transform.

        Assumptions:
            Missing/invalid values default to zero before scaling.
        """
        ds = union_children(self.children, by_name=False)

        def apply_ratio(batch: pd.DataFrame) -> pd.DataFrame:
            """Apply per-batch scaling/capping to duplicate counts.

            Args:
                batch: Source pandas batch.

            Returns:
                Batch with scaled `col` values.

            Side effects:
                None.

            Assumptions:
                Capping is required to avoid extreme oversampling multipliers.
            """
            out = batch.copy()
            out[self.col] = (
                pd.to_numeric(out[self.col], errors="coerce").fillna(0) * self.global_sample_rate
            ).clip(upper=self.max_sample)
            return out

        return map_batches_tuned(ds, self.runtime, apply_ratio, batch_format="pandas")


@register("Sampler")
class Sampler(PipelineNode):
    """Stochastically round duplicate weights and drop zero-sample rows.

    Inputs/outputs:
        Reads child dataset(s) and returns dataset with integer sample counts.

    Side effects:
        Uses NumPy random draws inside batch transforms.

    Assumptions:
        Fractional counts represent expected samples per row as used by
        quality-based selective repetition recipes.
    """

    def __init__(
        self,
        runtime,
        child_configs: list = None,
        col: str = "duplicate_count",
    ):
        """Configure stochastic sampling column.

        Args:
            runtime: Shared runtime config.
            child_configs: Upstream node configs.
            col: Count/probability column to sample.

        Returns:
            None.

        Side effects:
            None.

        Assumptions:
            Input values are non-negative expected counts.
            For duplicate_count workflows this column already encodes desired
            repeat frequency after any phase-level scaling.
        """
        super().__init__(runtime, child_configs)
        self.col = col

    def run(self):
        """Execute stochastic rounding-based sampling transform.

        Inputs/outputs:
            Reads child dataset(s) and returns sampled dataset.

        Side effects:
            Runs random batch transform.

        Assumptions:
            Rows with sampled count <= 0 should be removed.
        """
        ds = union_children(self.children, by_name=False)
        return self.sample(ds)

    def sample(self, ds):
        """Convert fractional sample counts into integer kept rows.

        Args:
            ds: Dataset containing sampling-count column.

        Returns:
            Dataset with updated integer counts and zero rows removed.

        Side effects:
            Executes random-number generation in Ray tasks.

        Assumptions:
            Each row's expected sample count is independent.
        """
        def apply_sample(batch: pd.DataFrame) -> pd.DataFrame:
            """Stochastically round counts within a batch.

            Args:
                batch: Source pandas batch.

            Returns:
                Batch filtered to rows with positive sampled count.

            Side effects:
                Uses pseudo-random draws via NumPy.

            Assumptions:
                `floor + Bernoulli(frac)` preserves expected sample value.
            """
            out = batch.copy()
            values = pd.to_numeric(out[self.col], errors="coerce").fillna(0)
            floors = np.floor(values)
            probs = values - floors
            random_draw = np.random.rand(len(out))
            sampled = floors + (random_draw < probs)
            out[self.col] = sampled.astype(int)
            out = out[out[self.col] > 0]
            return out

        return map_batches_tuned(ds, self.runtime, apply_sample, batch_format="pandas")


@register("Flatten")
class Flatten(PipelineNode):
    """Expand rows by integer repetition counts.

    Inputs/outputs:
        Reads child dataset(s) and returns expanded dataset without count column.

    Side effects:
        Materializes dataset to pandas for row repetition.

    Assumptions:
        Count column contains non-negative repeat counts.
        Intended for deterministic expansion when integer repetition is desired.
    """

    def __init__(
        self,
        runtime,
        child_configs: list = None,
        col: str = "duplicate_count",
    ):
        """Configure row-repeat flattening behavior.

        Args:
            runtime: Shared runtime config.
            child_configs: Upstream node configs.
            col: Repetition count column.

        Returns:
            None.

        Side effects:
            None.

        Assumptions:
            Repetition counts are integer-like and clipped at zero minimum.
            This is the deterministic companion to `Sampler`.
        """
        super().__init__(runtime, child_configs)
        self.col = col

    def run(self):
        """Materialize and expand rows according to repetition counts.

        Inputs/outputs:
            Reads child dataset(s) and returns expanded dataset.

        Side effects:
            Converts full dataset to pandas and back.

        Assumptions:
            Expansion can be memory intensive for large repeat totals.
            Used only when explicit repeated rows are required.
        """
        ds = union_children(self.children, by_name=False)
        return self.flatten(ds)

    def flatten(self, ds):
        """Repeat each row `col` times and drop the count column.

        Args:
            ds: Dataset to expand.

        Returns:
            Expanded dataset without repetition column.

        Side effects:
            Materializes full dataset to pandas.

        Assumptions:
            Empty input should pass through unchanged.
        """
        pdf = ds.to_pandas()
        if pdf.empty:
            return ds

        repeats = pd.to_numeric(pdf[self.col], errors="coerce").fillna(0).astype(int)
        repeats = repeats.clip(lower=0)
        non_repeat_cols = [c for c in pdf.columns if c != self.col]

        expanded = pdf.loc[pdf.index.repeat(repeats)][non_repeat_cols].reset_index(drop=True)
        return dataset_from_pandas(expanded)


@register("GroupFlatten")
class GroupFlatten(PipelineNode):
    """Flatten list-valued group columns into one row per element index.

    Inputs/outputs:
        Reads child dataset(s) and returns expanded dataset with extracted fields.

    Side effects:
        Materializes full dataset to pandas for row-wise expansion.

    Assumptions:
        Array columns are aligned by element index; shortest length per row wins.
    """

    def __init__(
        self,
        runtime,
        child_configs: list = None,
        cols: list = None,
        sub_cols: list = None,
        output_cols: list = None,
    ):
        """Configure grouped list flattening and output field extraction.

        Args:
            runtime: Shared runtime config.
            child_configs: Upstream node configs.
            cols: Source list columns to iterate.
            sub_cols: Keys/fields to extract per list item.
            output_cols: Destination column names for extracted values.

        Returns:
            None.

        Side effects:
            None.

        Assumptions:
            `sub_cols` and `output_cols` are positionally aligned.
        """
        super().__init__(runtime, child_configs)
        assert cols is not None and len(cols) >= 1, (
            "GroupFlatten requires at least one array column."
        )
        self.cols = cols
        self.sub_cols = sub_cols if sub_cols is not None else cols
        self.output_cols = output_cols if output_cols is not None else self.sub_cols
        assert len(self.sub_cols) == len(self.output_cols), (
            "sub_cols and output_cols must have the same length."
        )

    def run(self):
        """Run grouped flattening over child dataset output.

        Inputs/outputs:
            Reads child dataset(s) and returns flattened dataset.

        Side effects:
            May materialize full dataset in pandas.

        Assumptions:
            Children are union-compatible by position.
        """
        ds = union_children(self.children, by_name=False)
        return self.flatten(ds)

    def flatten(self, ds):
        """Expand list-valued row fields into multiple scalar rows.

        Args:
            ds: Dataset containing list-valued source columns.

        Returns:
            Expanded dataset with extracted per-element values.

        Side effects:
            Materializes full dataset to pandas.

        Assumptions:
            Rows missing list data are skipped.
        """
        pdf = ds.to_pandas()
        if pdf.empty:
            return ds

        # NOTE(readability): Row-wise expansion is intentionally eager because
        # each output row may derive from multiple list-valued source columns.
        expanded_rows = []
        for _, row in pdf.iterrows():
            arrays = [row.get(c) for c in self.cols]
            if not arrays or any(a is None for a in arrays):
                continue
            min_len = min(len(a) for a in arrays)
            for i in range(min_len):
                new_row = row.to_dict()
                for idx, sub_col in enumerate(self.sub_cols):
                    out_col = self.output_cols[idx]
                    source_col = self.cols[idx] if idx < len(self.cols) else self.cols[0]
                    value = row.get(source_col)
                    if isinstance(value, (list, tuple)) and i < len(value):
                        item = value[i]
                        if isinstance(item, dict):
                            new_row[out_col] = item.get(sub_col)
                        else:
                            new_row[out_col] = item
                    else:
                        new_row[out_col] = None
                expanded_rows.append(new_row)

        if not expanded_rows:
            return dataset_from_pandas(pd.DataFrame(columns=pdf.columns))

        out = pd.DataFrame(expanded_rows)
        return dataset_from_pandas(out)
