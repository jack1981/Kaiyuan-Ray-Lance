from typing import List, Union

import numpy as np

from datafiner.base import PipelineNode
from datafiner.dataset_utils import dataset_from_pandas, union_children
from datafiner.register import register


@register("Reorder")
class Reorder(PipelineNode):
    """
    Reorder a dataset by one or multiple score columns.
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
        pdf = ds.to_pandas()
        if pdf.empty:
            return ds

        sort_cols = []
        sort_ascending = []

        if self.folding > 1:
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
        ds = union_children(self.children, by_name=False)
        return self.reorder(ds)
