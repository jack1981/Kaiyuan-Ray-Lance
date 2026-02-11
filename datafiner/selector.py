"""Row selection node supporting deterministic and random sampling modes.

This node currently materializes datasets to pandas and applies selection logic
there for simplicity/compatibility with historical behavior.
Typical usage includes constructing top-ratio partitions from score-sorted data
for ablations similar to quantile/top-k dataset experiments.
"""

import pandas as pd

from datafiner.base import PipelineNode
from datafiner.dataset_utils import dataset_from_pandas, union_children
from datafiner.register import register


@register("Selector")
class Selector(PipelineNode):
    """
    Selects a subset of rows from a dataset.
    """

    def __init__(
        self,
        runtime,
        num_rows: int = None,
        selection_ratio: float = None,
        method: str = "random",
        score_col: str = None,
        approximate: bool = False,
        num_partitions: int = 32,
        seed: int = None,
        child_configs: list = None,
    ) -> None:
        """Configure row selection strategy and target size controls.

        Args:
            runtime: Shared runtime config.
            num_rows: Target row count to keep.
            selection_ratio: Fraction of rows to keep (alternative to num_rows).
            method: Selection method (`random`, `head`, `tail`, etc.).
            score_col: Score column used by `approximate` method.
            approximate: Whether approximate score-based mode is enabled.
            num_partitions: Reserved compatibility knob for approximate mode.
            seed: Optional RNG seed for random sampling.
            child_configs: Upstream node configs.

        Returns:
            None.

        Side effects:
            None during initialization.

        Assumptions:
            Exactly one of `num_rows` or `selection_ratio` is provided.
        """
        super().__init__(runtime, child_configs)

        valid_methods = ["random", "head", "tail", "depr_fast_random", "fast_random"]
        if approximate:
            valid_methods.append("approximate")
            if not score_col:
                raise ValueError(
                    "When 'approximate' is True, 'score_col' must be provided."
                )
            if num_partitions <= 0:
                raise ValueError(
                    "'num_partitions' must be a positive integer when 'approximate' is True."
                )

        if method not in valid_methods:
            raise ValueError(
                f"Method '{method}' is not supported. Choose one of {valid_methods}."
            )

        if num_rows is None and selection_ratio is None:
            raise ValueError("Either 'num_rows' or 'selection_ratio' must be provided.")

        if num_rows is not None and selection_ratio is not None:
            raise ValueError(
                "Only one of 'num_rows' or 'selection_ratio' should be provided, not both."
            )

        if num_rows is not None and (not isinstance(num_rows, int) or num_rows <= 0):
            raise ValueError("'num_rows' must be a positive integer.")

        if selection_ratio is not None and (
            not isinstance(selection_ratio, float) or not (0 < selection_ratio <= 1.0)
        ):
            raise ValueError("'selection_ratio' must be a float in the range (0, 1].")

        self.num_rows = num_rows
        self.selection_ratio = selection_ratio
        self.method = method
        self.score_col = score_col
        self.approximate = approximate
        self.num_partitions = num_partitions
        self.seed = seed

    def run(self):
        """Execute configured row-selection method on child data.

        Inputs/outputs:
            Reads child dataset(s) and returns selected subset dataset.

        Side effects:
            Materializes source data to pandas for selection operations.

        Assumptions:
            Selection semantics depend on pandas ordering at materialization time.
        """
        ds = union_children(self.children, by_name=False)
        return self.select(ds)

    def select(self, ds):
        """Select rows from dataset according to configured method and targets.

        Args:
            ds: Source dataset.

        Returns:
            Selected dataset.

        Side effects:
            Converts dataset to pandas and prints method info.

        Assumptions:
            `tail` mode keeps rows from index `target_rows` onward to match
            existing implementation behavior.
        """
        print(f"Sampling rows using '{self.method}' method.")
        pdf = ds.to_pandas()

        if pdf.empty:
            return ds

        target_rows = self.num_rows
        if target_rows is None and self.selection_ratio is not None:
            target_rows = max(1, int(len(pdf) * self.selection_ratio))

        if self.method == "head":
            result = pdf.head(target_rows)

        elif self.method == "tail":
            result = pdf.iloc[target_rows:]

        elif self.method in {"random", "depr_fast_random", "fast_random"}:
            if self.selection_ratio is not None:
                result = pdf.sample(frac=self.selection_ratio, random_state=self.seed)
            else:
                n = min(target_rows, len(pdf))
                result = pdf.sample(n=n, random_state=self.seed)

        elif self.method == "approximate":
            if self.score_col not in pdf.columns:
                raise ValueError(f"score_col '{self.score_col}' not found in data.")
            ascending = False
            ranked = pdf.sort_values(by=self.score_col, ascending=ascending)
            n = min(target_rows, len(ranked))
            result = ranked.head(n)

        else:
            raise ValueError(f"Unsupported selection method: {self.method}")

        return dataset_from_pandas(result.reset_index(drop=True))
