"""Ranking/quantile enrichment node for score-based datasets.

This node sorts rows by one score column and appends deterministic rank and/or
quantile fields.
It is primarily useful for quality-stratified analysis, including workflows
similar to the report's quantile benchmarking procedure.
"""

import numpy as np

from datafiner.base import PipelineNode
from datafiner.dataset_utils import dataset_from_pandas
from datafiner.register import register


@register("AddRankQuantile")
class AddRankQuantile(PipelineNode):
    """Add rank and quantile columns derived from a score column.

    Inputs/outputs:
        Reads first child dataset and returns sorted dataset with extra columns.

    Side effects:
        Materializes dataset to pandas for sorting and column computation.

    Assumptions:
        Score column exists and is sortable by pandas; quantile buckets are
        intended for dataset-quality analysis similar to quantile benchmarking.
    """

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
        """Configure score column and output enrichment toggles.

        Args:
            runtime: Shared runtime config.
            score_col: Column used for ranking.
            add_rank: Whether to add `rank` column.
            add_quantile: Whether to add `quantile` column.
            ascending: Sort order for rank assignment.
            require_sorted: Compatibility flag retained for API parity.
            child_configs: Upstream node configs.

        Returns:
            None.

        Side effects:
            None.

        Assumptions:
            At least one of `add_rank`/`add_quantile` is True.
        """
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
        """Run rank/quantile enrichment on first child dataset.

        Inputs/outputs:
            Reads first child dataset and returns enriched dataset.

        Side effects:
            Delegates to pandas-materializing helper.

        Assumptions:
            Node intentionally uses only the first child.
        """
        ds = self.children[0].run()
        return self._add_rank_and_quantile(ds)

    def _add_rank_and_quantile(self, ds):
        """Sort dataset and append rank/quantile columns as configured.

        Args:
            ds: Source dataset.

        Returns:
            Enriched dataset.

        Side effects:
            Materializes full dataset to pandas.

        Assumptions:
            Quantile is computed as `rank / N` with ranks starting at 1.
        """
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
