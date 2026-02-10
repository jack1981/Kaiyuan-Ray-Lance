import pandas as pd

from datafiner.base import PipelineNode
from datafiner.dataset_utils import dataset_from_pandas
from datafiner.register import register


def add_monotonic_id(df: pd.DataFrame, id_col_name: str = "row_id") -> pd.DataFrame:
    """Adds a monotonically increasing ID column to a pandas DataFrame."""
    with_id = df.copy()
    with_id[id_col_name] = range(len(with_id))
    return with_id


@register("Joiner")
class Joiner(PipelineNode):
    """Joins multiple DataFrames together based on a generated row ID."""

    def __init__(
        self,
        runtime,
        join_type: str = "inner",
        child_configs: list = None,
    ) -> None:
        super().__init__(runtime, child_configs)
        self.join_type = join_type
        self.join_key = "_temp_join_id"

    def run(self):
        # Ensure there are at least two children to join.
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
