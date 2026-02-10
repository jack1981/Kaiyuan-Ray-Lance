from typing import List, Union

import numpy as np

from datafiner.base import PipelineNode
from datafiner.dataset_utils import dataset_from_pandas
from datafiner.register import register


@register("InterleavedReorder")
class InterleavedReorder(PipelineNode):
    """
    Stratified interleaving reorder implemented with pandas on top of Ray Data.
    """

    def __init__(
        self,
        runtime,
        group_col: str,
        type_num: int,
        score_cols: Union[str, List[str]],
        ascending: bool = True,
        perturb: int = 0,
        child_configs: list = None,
    ) -> None:
        super().__init__(runtime, child_configs)
        self.group_col = group_col
        self.ascending = ascending
        self.perturb = perturb

        if isinstance(score_cols, str):
            self.score_cols = [score_cols]
        else:
            self.score_cols = score_cols

        if type_num <= 0:
            raise ValueError("'type_num' must be a positive integer.")
        if len(self.score_cols) != type_num:
            raise ValueError(
                f"The number of 'score_cols' ({len(self.score_cols)}) must match "
                f"the 'type_num' ({type_num})."
            )

    def reorder(self, ds):
        pdf = ds.to_pandas()
        if pdf.empty:
            return ds

        uuid_col = "__uuid__"
        random_key_col = "__random_key__"
        sort_key_col = "__group_sort_key__"
        rank_col = "__group_rank__"
        group_count_col = "__group_count__"
        rescaled_rank_col = "__rescaled_rank__"

        work = pdf.copy().reset_index(drop=True)
        work[uuid_col] = np.arange(len(work))
        work[random_key_col] = np.random.rand(len(work))

        def pick_sort_key(row):
            group = int(row[self.group_col])
            if group < 0 or group >= len(self.score_cols):
                return None
            col_name = self.score_cols[group]
            if col_name == "__random__":
                return row[random_key_col]
            return row.get(col_name)

        work[sort_key_col] = work.apply(pick_sort_key, axis=1)

        total_count = len(work)
        work[group_count_col] = work.groupby(self.group_col)[self.group_col].transform("count")

        work[rank_col] = (
            work.groupby(self.group_col)[sort_key_col]
            .rank(method="first", ascending=self.ascending)
            .astype(float)
        )

        work[rescaled_rank_col] = (work[rank_col] / work[group_count_col]) * total_count

        if self.perturb > 0:
            work[rescaled_rank_col] = work[rescaled_rank_col] + (
                self.perturb * np.random.rand(len(work))
            )

        sorted_df = work.sort_values(
            by=[rescaled_rank_col, random_key_col], ascending=[True, True]
        ).reset_index(drop=True)

        sorted_df = sorted_df.drop(columns=[uuid_col, random_key_col, sort_key_col, rank_col, group_count_col, rescaled_rank_col])
        return dataset_from_pandas(sorted_df)

    def run(self):
        if not self.children:
            raise ValueError(f"{self.__class__.__name__} node has no child.")
        ds = self.children[0].run()
        return self.reorder(ds)
