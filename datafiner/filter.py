import operator
from enum import Enum

import pandas as pd

from datafiner.base import PipelineNode
from datafiner.dataset_utils import dataset_from_pandas, union_children
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
        self.symbol = symbol
        self.func = func

    @classmethod
    def from_str(cls, s: str):
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
        super().__init__(runtime, child_configs)
        if column is None:
            raise ValueError("The 'column' argument must be specified for Filter.")
        self.column = column
        self.comp_op = ComparisonOperator.from_str(comparison)
        self.threshold = threshold

    def run(self):
        ds = union_children(self.children, by_name=False)

        def filter_batch(batch: pd.DataFrame) -> pd.DataFrame:
            values = pd.to_numeric(batch[self.column], errors="coerce")
            mask = self.comp_op.func(values, self.threshold)
            return batch[mask.fillna(False)]

        return ds.map_batches(filter_batch, batch_format="pandas")


@register("FilterByRatio")
class FilterByRatio(PipelineNode):
    """
    Filter a dataset to keep a target ratio based on a score column.
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
        ds = union_children(self.children, by_name=False)
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
