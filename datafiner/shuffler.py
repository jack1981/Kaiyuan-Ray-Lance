from datafiner.base import PipelineNode
from datafiner.register import register
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql import functions as F


@register("Shuffler")
class Shuffler(PipelineNode):
    """
    A pipeline node to globally shuffle all rows of a DataFrame.
    """

    def __init__(
        self,
        spark: SparkSession,
        child_configs: list = None,
    ):
        """
        Initializes the Shuffle node.

        Args:
            spark (SparkSession): The Spark session object.
            child_configs (list, optional): List of child node configurations. Defaults to None.
        """
        super().__init__(spark, child_configs)

    def run(self) -> DataFrame:
        """
        Executes the global shuffle.

        Returns:
            DataFrame: The shuffled DataFrame.
        """
        df = self.children[0].run()
        if len(self.children) > 1:
            for child in self.children[1:]:
                df = df.union(child.run())

        return df.orderBy(F.rand())
