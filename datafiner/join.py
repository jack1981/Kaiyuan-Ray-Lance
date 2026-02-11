"""Pandas-based join node for combining multiple child datasets by row index.

This module adds temporary monotonic row ids per child and joins on that key,
matching historical position-based join semantics.
"""

import pandas as pd

from datafiner.base import PipelineNode
from datafiner.dataset_utils import dataset_from_pandas
from datafiner.register import register


def add_monotonic_id(df: pd.DataFrame, id_col_name: str = "row_id") -> pd.DataFrame:
    """Add a zero-based monotonic id column to a DataFrame copy.

    Args:
        df: Input pandas DataFrame.
        id_col_name: Name of generated id column.

    Returns:
        Copied DataFrame with id column.

    Side effects:
        None.

    Assumptions:
        Existing column with same name can be overwritten in the copy.
    """
    with_id = df.copy()
    with_id[id_col_name] = range(len(with_id))
    return with_id


@register("Joiner")
class Joiner(PipelineNode):
    """Join multiple child datasets on generated monotonic row ids.

    Inputs/outputs:
        Reads child datasets, converts to pandas, and returns joined dataset.

    Side effects:
        Materializes all child datasets to pandas and performs pandas merges.

    Assumptions:
        Join key represents row order, not semantic business keys.
    """

    def __init__(
        self,
        runtime,
        join_type: str = "inner",
        child_configs: list = None,
    ) -> None:
        """Configure join type and initialize temporary join key name.

        Args:
            runtime: Shared runtime config.
            join_type: Pandas merge `how` mode.
            child_configs: Upstream node configs.

        Returns:
            None.

        Side effects:
            None.

        Assumptions:
            At least two children are provided when run.
        """
        super().__init__(runtime, child_configs)
        self.join_type = join_type
        self.join_key = "_temp_join_id"

    def run(self):
        """Join child outputs using temporary row-id columns.

        Inputs/outputs:
            Reads all child datasets and returns joined dataset.

        Side effects:
            Materializes child datasets to pandas and performs merge operations.

        Assumptions:
            Row-order alignment across children is intentional.
        """
        # NOTE(readability): This operator is intentionally eager because join
        # semantics are implemented with pandas row-id alignment.
        if len(self.children) < 2:
            raise ValueError("Joiner node requires at least two child nodes.")

        # Process the first child and add a join key.
        base_df = add_monotonic_id(self.children[0].run().to_pandas(), self.join_key)

        # Iteratively join the rest of the children.
        for child in self.children[1:]:
            df_to_join = add_monotonic_id(child.run().to_pandas(), self.join_key)
            base_df = base_df.merge(df_to_join, on=self.join_key, how=self.join_type)

        # Drop the temporary join key before returning.
        return dataset_from_pandas(base_df.drop(columns=[self.join_key]))
