"""Filtering nodes for threshold and ratio-based dataset pruning.

These nodes operate on numeric score columns using pandas batch transforms or
full-data quantile calculations when ratio targets are requested.
This aligns with the report's quality-metric based top-k / quantile partition
selection strategy used before downstream mixing and repetition.
"""

import operator
from enum import Enum

import pandas as pd

from datafiner.base import PipelineNode
from datafiner.dataset_utils import dataset_from_pandas, map_batches_tuned, union_children
from datafiner.register import register


class ComparisonOperator(Enum):
    """
    Enumeration for comparison operators.
    """

    LARGER = (">", operator.gt)
    SMALLER = ("<", operator.lt)
    LARGER_OR_EQUAL = (">=", operator.ge)
    SMALLER_OR_EQUAL = ("<=", operator.le)
    EQUAL = ("==", operator.eq)

    def __init__(self, symbol, func):
        """Store both symbolic and callable form of a comparison operator.

        Inputs/outputs:
            Accepts operator symbol and comparison function; returns None.

        Side effects:
            None.

        Assumptions:
            Enum members are immutable after construction.
        """
        self.symbol = symbol
        self.func = func

    @classmethod
    def from_str(cls, s: str):
        """Parse human-friendly operator text into enum values.

        Args:
            s: Operator name or symbol.

        Returns:
            Matching `ComparisonOperator`.

        Side effects:
            None.

        Assumptions:
            Comparison names are case-insensitive and trimmed.
        """
        s_lower = s.lower().strip()
        if s_lower in ("larger", ">"):
            return cls.LARGER
        if s_lower in ("smaller", "<"):
            return cls.SMALLER
        if s_lower in ("larger_or_equal", ">="):
            return cls.LARGER_OR_EQUAL
        if s_lower in ("smaller_or_equal", "<="):
            return cls.SMALLER_OR_EQUAL
        if s_lower in ("equal", "=="):
            return cls.EQUAL
        raise ValueError(f"Unsupported comparison operator: {s}")


@register("Filter")
class Filter(PipelineNode):
    """
    Filter a dataset based on a static threshold.
    """

    def __init__(
        self,
        runtime,
        child_configs: list = None,
        column: str = None,
        comparison: str = "larger",
        threshold: float = 0.0,
    ):
        """Configure static-threshold filtering.

        Args:
            runtime: Shared runtime config.
            child_configs: Upstream node configs.
            column: Numeric column to compare.
            comparison: Comparison operator name/symbol.
            threshold: Numeric threshold value.

        Returns:
            None.

        Side effects:
            None.

        Assumptions:
            Non-numeric values are coerced to NaN and treated as non-matching.
        """
        super().__init__(runtime, child_configs)
        if column is None:
            raise ValueError("The 'column' argument must be specified for Filter.")
        self.column = column
        self.comp_op = ComparisonOperator.from_str(comparison)
        self.threshold = threshold

    def run(self):
        """Filter rows using configured comparator and threshold.

        Inputs/outputs:
            Reads child dataset(s) and returns filtered dataset.

        Side effects:
            Executes `map_batches` across all blocks.

        Assumptions:
            Batch schemas include the configured `column`.
        """
        ds = union_children(self.children, by_name=False)

        def filter_batch(batch: pd.DataFrame) -> pd.DataFrame:
            """Apply threshold predicate to one pandas batch.

            Args:
                batch: Source pandas batch.

            Returns:
                Filtered batch.

            Side effects:
                None.

            Assumptions:
                Missing/invalid numeric values should be dropped.
            """
            values = pd.to_numeric(batch[self.column], errors="coerce")
            mask = self.comp_op.func(values, self.threshold)
            return batch[mask.fillna(False)]

        return map_batches_tuned(ds, self.runtime, filter_batch, batch_format="pandas")


@register("FilterByRatio")
class FilterByRatio(PipelineNode):
    """
    Filter a dataset to keep a target ratio based on a score column.

    This corresponds to top-k style quality filtering used in phase-wise data
    refinement (for example keeping top 50/30/10 percent partitions).
    """

    def __init__(
        self,
        runtime,
        child_configs: list = None,
        column: str = None,
        comparison: str = "larger",
        keep_ratio: float = 0.1,
        quantile_error: float = 1e-4,
    ):
        """Configure quantile-based keep-ratio filtering.

        Args:
            runtime: Shared runtime config.
            child_configs: Upstream node configs.
            column: Numeric score column.
            comparison: `larger` keeps top scores; `smaller` keeps bottom.
            keep_ratio: Fraction of rows to keep.
            quantile_error: Reserved compatibility arg (not currently used).

        Returns:
            None.

        Side effects:
            None.

        Assumptions:
            Full dataset materialization to pandas is acceptable for this node.
        """
        super().__init__(runtime, child_configs)
        if not (0.0 <= keep_ratio <= 1.0):
            raise ValueError("keep_ratio must be between 0.0 and 1.0")
        if column is None:
            raise ValueError("The 'column' argument must be specified for FilterByRatio.")
        self.column = column
        self.comp_op = ComparisonOperator.from_str(comparison)
        self.keep_ratio = keep_ratio
        self.quantile_error = quantile_error

    def run(self):
        """Keep approximately `keep_ratio` rows based on score quantile threshold.

        Inputs/outputs:
            Reads child dataset(s) and returns filtered dataset.

        Side effects:
            Materializes full dataset to pandas and computes quantiles.

        Assumptions:
            Supports only `larger` or `smaller` comparisons for ratio filtering,
            mapping directly to keep-highest or keep-lowest partitions.
        """
        ds = union_children(self.children, by_name=False)
        # NOTE(readability): This operator is intentionally eager because quantile
        # thresholding needs global score distribution.
        pdf = ds.to_pandas()
        if pdf.empty:
            return ds

        values = pd.to_numeric(pdf[self.column], errors="coerce")

        if self.comp_op == ComparisonOperator.LARGER:
            quantile = 1.0 - self.keep_ratio
        elif self.comp_op == ComparisonOperator.SMALLER:
            quantile = self.keep_ratio
        else:
            raise ValueError("FilterByRatio only supports 'larger' or 'smaller'.")

        threshold = values.quantile(quantile)
        self.threshold = float(threshold)
        mask = self.comp_op.func(values, threshold)
        filtered = pdf[mask.fillna(False)].reset_index(drop=True)
        return dataset_from_pandas(filtered)
