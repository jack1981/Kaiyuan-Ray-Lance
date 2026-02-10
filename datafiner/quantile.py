import numpy as np

from datafiner.base import PipelineNode
from datafiner.dataset_utils import dataset_from_pandas
from datafiner.register import register


@register("AddRankQuantile")
class AddRankQuantile(PipelineNode):
    def __init__(
        self,
        runtime,
        score_col: str,
        add_rank: bool = True,
        add_quantile: bool = True,
        ascending: bool = False,
        require_sorted: bool = False,
        child_configs: list = None,
    ) -> None:
        super().__init__(runtime, child_configs)

        if not score_col or not isinstance(score_col, str):
            raise ValueError("'score_col' must be a non-empty string.")
        if not add_rank and not add_quantile:
            raise ValueError("At least one of 'add_rank' or 'add_quantile' must be True.")

        self.score_col = score_col
        self.add_rank = add_rank
        self.add_quantile = add_quantile
        self.ascending = ascending
        self.require_sorted = require_sorted

    def run(self):
        ds = self.children[0].run()
        return self._add_rank_and_quantile(ds)

    def _add_rank_and_quantile(self, ds):
        if not self.add_rank and not self.add_quantile:
            return ds

        print(
            f"Adding rank/quantile based on '{self.score_col}' in {'ascending' if self.ascending else 'descending'} order."
        )

        pdf = ds.to_pandas()
        if pdf.empty:
            return ds

        sorted_pdf = pdf.sort_values(
            by=self.score_col, ascending=self.ascending
        ).reset_index(drop=True)

        rank_values = np.arange(1, len(sorted_pdf) + 1, dtype=float)

        if self.add_rank:
            sorted_pdf["rank"] = rank_values

        if self.add_quantile:
            sorted_pdf["quantile"] = rank_values / max(len(sorted_pdf), 1)

        return dataset_from_pandas(sorted_pdf)
