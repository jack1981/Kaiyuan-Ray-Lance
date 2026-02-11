"""Group-aware interleaving reorder node for multi-type score balancing.

This node computes per-group ranks and rescales them into a single ordering
space to interleave groups while respecting group-local score preferences.
The rescale-and-interleave design follows the same intent as Algorithm 1 in the
PCMind-2.1 report (within-dataset ordering + global proportional interleaving).
"""

from typing import List, Union

import numpy as np

from datafiner.base import PipelineNode
from datafiner.dataset_utils import dataset_from_pandas
from datafiner.register import register


@register("InterleavedReorder")
class InterleavedReorder(PipelineNode):
    """
    Stratified interleaving reorder implemented with pandas on top of Ray Data.

    The algorithm mirrors the report's curriculum-construction idea: preserve
    within-group ordering while interleaving groups proportionally.
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
        """Configure grouped interleaving order behavior.

        Args:
            runtime: Shared runtime config.
            group_col: Integer-like group id column.
            type_num: Expected number of groups/types.
            score_cols: Per-group score columns (or `__random__` placeholders).
            ascending: Sort direction within each group score.
            perturb: Random perturbation scale applied to rescaled rank.
            child_configs: Upstream node configs.

        Returns:
            None.

        Side effects:
            None during initialization.

        Assumptions:
            Group ids are in `[0, type_num)` and align with `score_cols`.
        """
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
        """Interleave groups by normalized within-group rank.

        Args:
            ds: Source dataset.

        Returns:
            Reordered dataset.

        Side effects:
            Materializes full dataset to pandas and uses random values for
            per-row keys and optional perturbation.

        Assumptions:
            Ranking is stable (`method="first"`) to preserve deterministic
            tie-breaking before optional random perturbation; this is important
            for reproducible curriculum ordering.
        """
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
            """Pick score key for one row based on row group id.

            Args:
                row: Pandas row with group id and candidate score columns.

            Returns:
                Selected numeric sort key or `None`.

            Side effects:
                None.

            Assumptions:
                Invalid group ids should fall back to `None` and naturally sort
                last/first depending on pandas behavior.
            """
            group = int(row[self.group_col])
            if group < 0 or group >= len(self.score_cols):
                return None
            col_name = self.score_cols[group]
            if col_name == "__random__":
                # NOTE(readability): The random-key fallback mirrors the report's
                # handling for datasets without explicit quality metrics.
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

        # NOTE(readability): Rescaling rank by group size projects each group's
        # local rank into a global space so groups are interleaved fairly.
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
        """Execute grouped reorder using the first child dataset.

        Inputs/outputs:
            Reads first child dataset and returns reordered dataset.

        Side effects:
            Raises when no child exists.

        Assumptions:
            Node intentionally uses only one child input.
        """
        if not self.children:
            raise ValueError(f"{self.__class__.__name__} node has no child.")
        ds = self.children[0].run()
        return self.reorder(ds)
