import os
import numpy as np
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql import functions as F, Window
from datafiner.base import PipelineNode
from datafiner.register import register
from pyspark.sql.types import DoubleType


@register("AddRankQuantile")
class AddRankQuantile(PipelineNode):
    def __init__(
        self,
        spark: SparkSession,
        score_col: str,
        add_rank: bool = True,
        add_quantile: bool = True,
        ascending: bool = False,
        require_sorted: bool = False,
        child_configs: list = None,
    ) -> None:
        """
        Pipeline node to add rank and/or quantile columns based on a score column.

        Args:
            spark: The SparkSession object.
            score_col: The name of the column to rank by.
            add_rank: If True, adds a 'rank' column.
            add_quantile: If True, adds a 'quantile' column.
            ascending: If True, ranks in ascending order.
            child_configs: Configuration for child nodes in the pipeline.
        """
        super().__init__(spark, child_configs)

        if not score_col or not isinstance(score_col, str):
            raise ValueError("'score_col' must be a non-empty string.")
        if not add_rank and not add_quantile:
            raise ValueError(
                "At least one of 'add_rank' or 'add_quantile' must be True."
            )

        self.score_col = score_col
        self.add_rank = add_rank
        self.add_quantile = add_quantile
        self.ascending = ascending
        self.require_sorted = require_sorted

    def run(self) -> DataFrame:
        """
        Executes the pipeline node's logic.
        """
        df = self.children[0].run()
        # The union logic is better handled by a dedicated 'UnionNode' if needed,
        # keeping this node's responsibility focused.
        return self._add_rank_and_quantile(df)

    def _add_rank_and_quantile(self, df: DataFrame) -> DataFrame:
        """
        Adds rank and quantile columns to the DataFrame efficiently using window functions.
        """
        if not self.add_rank and not self.add_quantile:
            return df

        print(
            f"Adding rank/quantile based on '{self.score_col}' in {'ascending' if self.ascending else 'descending'} order."
        )

        # 1. Define the window specification for ordering.
        # This single window will be used for all calculations.
        order_col = (
            F.col(self.score_col).asc()
            if self.ascending
            else F.col(self.score_col).desc()
        )
        # The window covers the entire DataFrame for global ranking.
        if self.require_sorted:
            df = df.orderBy(order_col)
        window = Window.orderBy(order_col)

        # 2. Calculate rank and total count in a single pass using the window.
        # This is the most efficient method as it avoids a separate .count() action.
        df_with_calcs = df.withColumn("rank", F.row_number().over(window))

        if self.add_quantile:
            # Calculate total count within the same window definition but over an empty partition
            # This is an efficient way to get the total count as a column
            total_count_window = (
                Window.partitionBy()
            )  # Empty partition means global count
            df_with_calcs = df_with_calcs.withColumn(
                "total_count", F.count("*").over(total_count_window)
            )

            # Calculate quantile
            df_with_calcs = df_with_calcs.withColumn(
                "quantile", F.col("rank") / F.col("total_count")
            ).drop("total_count")  # Drop the intermediate total_count column

        # 3. Conditionally drop the rank column if not needed.
        if not self.add_rank:
            df_with_calcs = df_with_calcs.drop("rank")
        else:
            # Ensure rank column is of type Double for consistency
            df_with_calcs = df_with_calcs.withColumn(
                "rank", F.col("rank").cast(DoubleType())
            )

        return df_with_calcs
