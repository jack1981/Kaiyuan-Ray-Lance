from datafiner.base import PipelineNode
from datafiner.register import register
from pyspark.sql import SparkSession, DataFrame


@register("Visualizer")
class Visualizer(PipelineNode):
    """
    A pipeline node to Visualize a specified number of rows from a DataFrame.
    """

    def __init__(
        self,
        spark: SparkSession,
        child_configs: list = None,
        num_rows: int = 20,
        vertical: bool = False,
    ):
        """
        Initializes the Visualize node.

        Args:
            spark (SparkSession): The Spark session object.
            child_configs (list, optional): List of child node configurations. Defaults to None.
            num_rows (int, optional): The number of rows to visualize. Defaults to 20.
            vertical (bool, optional): Whether to display the output vertically. Defaults to False.
        """
        super().__init__(spark, child_configs)
        self.num_rows = num_rows
        self.vertical = vertical

    def run(self) -> DataFrame:
        """
        Executes the visualization process.

        Returns:
            DataFrame: The input DataFrame, passed through without modification.
        """
        df = self.children[0].run()
        if len(self.children) > 1:
            for child in self.children[1:]:
                df = df.union(child.run())

        print(f"Visualizing the first {self.num_rows} rows:")
        df.show(n=self.num_rows, vertical=self.vertical)

        return df
