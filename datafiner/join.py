from pyspark.sql import SparkSession, DataFrame
from pyspark.sql import functions as F
from datafiner.base import PipelineNode
from datafiner.register import register


def add_monotonic_id(df: DataFrame, id_col_name: str = "row_id") -> DataFrame:
    """Adds a monotonically increasing ID column to a DataFrame."""
    return df.withColumn(id_col_name, F.monotonically_increasing_id())


@register("Joiner")
class Joiner(PipelineNode):
    """Joins multiple DataFrames together based on a generated row ID."""

    def __init__(
        self,
        spark: SparkSession,
        join_type: str = "inner",
        child_configs: list = None,
    ) -> None:
        super().__init__(spark, child_configs)
        self.join_type = join_type
        self.join_key = "_temp_join_id"

    def run(self) -> DataFrame:
        # Ensure there are at least two children to join.
        if len(self.children) < 2:
            raise ValueError("Joiner node requires at least two child nodes.")

        # Process the first child and add a join key.
        base_df = add_monotonic_id(self.children[0].run(), self.join_key)

        # Iteratively join the rest of the children.
        for child in self.children[1:]:
            df_to_join = add_monotonic_id(child.run(), self.join_key)
            base_df = base_df.join(df_to_join, on=self.join_key, how=self.join_type)

        # Drop the temporary join key before returning.
        return base_df.drop(self.join_key)
