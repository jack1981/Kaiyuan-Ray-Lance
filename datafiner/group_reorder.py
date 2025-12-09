from pyspark.sql import SparkSession, DataFrame
from pyspark.sql import functions as F
from pyspark.sql.window import Window
from typing import List, Union
import logging

from datafiner.register import register
from datafiner.base import PipelineNode


@register("InterleavedReorder")
class InterleavedReorder(PipelineNode):
    """
    Implements a stratified reordering algorithm using a split-calculate-join
    optimization strategy to reduce shuffle overhead.

    This node reorders data by:
    1. Grouping rows by a specified column
    2. Ranking rows within each group based on score columns
    3. Rescaling ranks proportionally across all groups
    4. Sorting the entire dataset by rescaled ranks

    This produces an interleaved ordering where items from different groups
    are mixed proportionally based on their within-group rankings.
    """

    def __init__(
        self,
        spark: SparkSession,
        group_col: str,
        type_num: int,
        score_cols: Union[str, List[str]],
        ascending: bool = True,
        perturb: int = 0,
        child_configs: list = None,
    ) -> None:
        """
        Args:
            spark: SparkSession instance
            group_col: Column name to group by
            type_num: Number of distinct group types
            score_cols: Column name(s) to use for scoring within each group.
                       Use "__random__" for random ordering within a group.
            ascending: Whether to sort scores in ascending order
            perturb: Random perturbation factor for rescaled ranks (0 = no perturbation)
            child_configs: Child pipeline configurations
        """
        super().__init__(spark, child_configs)
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

    def reorder(self, df: DataFrame) -> DataFrame:
        """
        Applies the reordering logic using an optimized split-join strategy.

        Algorithm:
        1. Add unique identifiers and random keys to each row
        2. Extract minimal columns needed for ranking (uuid, group, score)
        3. Calculate within-group ranks using window functions
        4. Rescale ranks proportionally to total dataset size
        5. Join rescaled ranks back to original data
        6. Sort by rescaled ranks to produce interleaved output

        Args:
            df: Input DataFrame

        Returns:
            Reordered DataFrame with temporary columns removed
        """

        # Define temporary column names
        UUID_COL = "__uuid__"
        RANDOM_KEY_COL = "__random_key__"
        SORT_KEY_COL = "__group_sort_key__"
        RANK_COL = "__group_rank__"
        GROUP_COUNT_COL = "__group_count__"
        RESCALED_RANK_COL = "__rescaled_rank__"

        # Step 1: Add UUID and random key to each row
        df_with_ids = df.withColumn(UUID_COL, F.expr("uuid()")).withColumn(
            RANDOM_KEY_COL, F.rand()
        )
        logging.info("Step 1: Added UUID and random key columns")

        # Step 2: Create lightweight DataFrame with only necessary columns
        # Build dynamic WHEN expression to select appropriate score column based on group
        when_expr = None
        for i, col_name in enumerate(self.score_cols):
            condition = F.col(self.group_col) == i

            if col_name == "__random__":
                value_expr = F.col(RANDOM_KEY_COL)
            else:
                value_expr = F.col(col_name)

            if when_expr is None:
                when_expr = F.when(condition, value_expr)
            else:
                when_expr = when_expr.when(condition, value_expr)

        score_df = df_with_ids.select(
            F.col(UUID_COL),
            F.col(self.group_col),
            when_expr.otherwise(None).alias(SORT_KEY_COL),
        )
        logging.info("Step 2: Created score DataFrame with minimal columns")

        # Step 3: Get total count for rescaling
        try:
            self.spark.sparkContext.setJobDescription(
                "InterleavedReorder - Calculate total count"
            )
            total_count = df.count()
            logging.info(f"Step 3: Total row count = {total_count}")
        finally:
            self.spark.sparkContext.setJobDescription(None)

        # Step 4: Calculate within-group ranks and rescale
        window_group_count = Window.partitionBy(self.group_col)
        sort_expr = (
            F.col(SORT_KEY_COL).asc() if self.ascending else F.col(SORT_KEY_COL).desc()
        )
        window_group_rank = Window.partitionBy(self.group_col).orderBy(sort_expr)

        rescaled_df = (
            score_df.withColumn(
                GROUP_COUNT_COL, F.count(F.lit(1)).over(window_group_count)
            )
            .withColumn(RANK_COL, F.row_number().over(window_group_rank))
            .withColumn(
                RESCALED_RANK_COL,
                (F.col(RANK_COL) / F.col(GROUP_COUNT_COL)) * F.lit(total_count),
            )
        )

        if self.perturb > 0:
            rescaled_df = rescaled_df.withColumn(
                RESCALED_RANK_COL,
                F.col(RESCALED_RANK_COL) + F.lit(self.perturb) * F.rand(),
            )

        logging.info("Step 4: Calculated and rescaled within-group ranks")

        # Step 5: Join rescaled ranks back to original DataFrame
        final_df = df_with_ids.join(
            rescaled_df.select(UUID_COL, RESCALED_RANK_COL), on=UUID_COL, how="inner"
        )
        logging.info("Step 5: Joined rescaled ranks to original data")

        # Step 6: Sort by rescaled rank to produce interleaved output
        sorted_df = final_df.orderBy(
            F.col(RESCALED_RANK_COL).asc(), F.col(RANDOM_KEY_COL).asc()
        )
        logging.info("Step 6: Final sort completed")

        # Clean up temporary columns
        return sorted_df.drop(UUID_COL, RANDOM_KEY_COL, RESCALED_RANK_COL)

    def run(self) -> DataFrame:
        """
        Executes the pipeline node to fetch data from children and reorder it.

        Returns:
            Reordered DataFrame

        Raises:
            ValueError: If no child nodes are configured
        """
        if not self.children:
            raise ValueError(f"{self.__class__.__name__} node has no child.")
        if len(self.children) > 1:
            logging.warning(
                f"{self.__class__.__name__} expects a single child, "
                f"but got {len(self.children)}. Only the first child will be processed."
            )

        df = self.children[0].run()
        return self.reorder(df)
