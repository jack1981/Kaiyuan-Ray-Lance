from pyspark.sql import SparkSession, DataFrame
from datafiner.base import PipelineNode
from datafiner.register import register
import pyspark.sql.functions as F


@register("Concat")
class Concat(PipelineNode):
    """
    A pipeline node to select a specific set of columns,
    dropping all others.
    """

    def __init__(
        self,
        spark: SparkSession,
        select_cols: list,
        output_col: str,
        child_configs: list = None,
    ):
        """
        Initializes the ColumnSelect node.

        Args:
            spark (SparkSession): The Spark session object.
            select_cols (list): A list of column names to keep.
            child_configs (list, optional): List of child node configurations.
        """
        super().__init__(spark, child_configs)
        if not isinstance(select_cols, list) or not select_cols:
            raise ValueError("'select_cols' must be a non-empty list.")
        self.select_cols = select_cols
        self.output_col = output_col

    def run(self):
        df = self.children[0].run()
        if len(self.children) > 1:
            for child in self.children[1:]:
                df = df.union(child.run())

        print(f"[ColumnSelect] Selecting columns: {self.select_cols}")
        return df.withColumn(self.output_col, F.concat(*self.select_cols))
