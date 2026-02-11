"""Sorting/reordering node for score-driven corpus ordering.

This implementation materializes to pandas and supports optional fold grouping
and randomized tiebreaking before converting back to Ray Dataset.
It maps naturally to within-dataset quality ordering steps used in curriculum
construction.
"""

from typing import List, Union

import numpy as np

from datafiner.base import PipelineNode
from datafiner.dataset_utils import dataset_from_pandas, union_children
from datafiner.register import register


@register("Reorder")
class Reorder(PipelineNode):
    """
    Reorder a dataset by one or multiple score columns.

    This operator represents within-dataset quality ordering before top-ratio
    filtering or global interleaving with other datasets.
    """

    def __init__(
        self,
        runtime,
        score_cols: Union[str, List[str]],
        ascending: Union[bool, List[bool]] = True,
        use_random_tiebreaker: bool = False,
        approximate: bool = False,
        num_partitions: int = 32,
        folding: int = 1,
        child_configs: list = None,
    ) -> None:
        """Configure sort keys and ordering behavior.

        Args:
            runtime: Shared runtime config.
            score_cols: One or more score columns to sort by.
            ascending: Bool or list of bools matching score columns.
            use_random_tiebreaker: Whether to add random tie-break key.
            approximate: Reserved compatibility flag for prior APIs.
            num_partitions: Reserved compatibility partition count.
            folding: Optional fold-id column count for interleaved sorting.
            child_configs: Upstream node configs.

        Returns:
            None.

        Side effects:
            None during initialization.

        Assumptions:
            Length of `ascending` must match number of score columns.
        """
        super().__init__(runtime, child_configs)

        if isinstance(score_cols, str):
            self.score_cols = [score_cols]
        elif isinstance(score_cols, list):
            self.score_cols = score_cols
        else:
            raise TypeError("'score_cols' must be a string or a list of strings.")

        if isinstance(ascending, bool):
            self.ascending = [ascending] * len(self.score_cols)
        elif isinstance(ascending, list):
            self.ascending = ascending
        else:
            raise TypeError("'ascending' must be a boolean or a list of booleans.")

        if len(self.score_cols) != len(self.ascending):
            raise ValueError(
                f"The number of score columns ({len(self.score_cols)}) must match "
                f"the number of ascending flags ({len(self.ascending)})."
            )

        if approximate and num_partitions <= 0:
            raise ValueError("'num_partitions' must be positive for approximate ordering.")

        self.use_random_tiebreaker = use_random_tiebreaker
        self.approximate = approximate
        self.num_partitions = num_partitions
        self.folding = folding

    def reorder(self, ds):
        """Sort dataset rows according to configured keys and optional helpers.

        Args:
            ds: Source dataset.

        Returns:
            Sorted dataset.

        Side effects:
            Materializes full dataset to pandas and uses NumPy random values when
            tie-breaker is enabled.

        Assumptions:
            Sorting in pandas should preserve deterministic ordering for identical
            inputs when random tiebreaking is disabled.
        """
        pdf = ds.to_pandas()
        if pdf.empty:
            return ds

        sort_cols = []
        sort_ascending = []

        if self.folding > 1:
            # NOTE(readability): Folding adds a deterministic round-robin key to
            # spread adjacent rows before score-based sorting.
            pdf["__fold_id__"] = np.arange(len(pdf)) % self.folding
            sort_cols.append("__fold_id__")
            sort_ascending.append(True)

        sort_cols.extend(self.score_cols)
        sort_ascending.extend(self.ascending)

        if self.use_random_tiebreaker:
            pdf["__random_tiebreaker__"] = np.random.rand(len(pdf))
            sort_cols.append("__random_tiebreaker__")
            sort_ascending.append(True)

        sorted_pdf = pdf.sort_values(by=sort_cols, ascending=sort_ascending).reset_index(
            drop=True
        )

        temp_cols = [c for c in ["__fold_id__", "__random_tiebreaker__"] if c in sorted_pdf.columns]
        if temp_cols:
            sorted_pdf = sorted_pdf.drop(columns=temp_cols)

        return dataset_from_pandas(sorted_pdf)

    def run(self):
        """Read child data and return reordered dataset output.

        Inputs/outputs:
            Reads child dataset(s) and returns sorted dataset.

        Side effects:
            Delegates to pandas-materializing reorder implementation.

        Assumptions:
            Child datasets are union-compatible by position.
        """
        ds = union_children(self.children, by_name=False)
        return self.reorder(ds)
