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
        ds = union_children(self.children, by_name=False)
        return self.select(ds)

    def select(self, ds):
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
