from pyspark.sql import SparkSession, DataFrame
from pyspark.sql import functions as F
from datafiner.base import PipelineNode
from datafiner.register import register
from typing import List, Union


@register("Reorder")
class Reorder(PipelineNode):
    """
    A pipeline node to reorder a DataFrame based on one or more columns.

    This node supports multi-column sorting with individual sort directions,
    approximate sorting for performance, and a random tie-breaker to prevent
    data skew on large datasets with many identical sort keys.
    """

    def __init__(
        self,
        spark: SparkSession,
        score_cols: Union[str, List[str]],
        ascending: Union[bool, List[bool]] = True,
        use_random_tiebreaker: bool = False,
        approximate: bool = False,
        num_partitions: int = 32,
        folding: int = 1,
        child_configs: list = None,
    ) -> None:
        """
        Initializes the Reorder node.

        Args:
            spark: The active Spark session.
            score_cols: A single column name or a list of column names to sort by.
            ascending: A single boolean (for all columns) or a list of booleans
                       specifying the sort direction for each corresponding score column.
            use_random_tiebreaker: If True, adds a random column as the last sort key
                                   to ensure a deterministic order and prevent data skew.
            approximate: If True, uses sortWithinPartitions for an approximate global order.
            num_partitions: Number of partitions for approximate ordering.
            folding: If > 1, folds the data into N groups to process them independently,
                     which can help manage memory.
            child_configs: Configuration for child nodes.
        """
        super().__init__(spark, child_configs)

        # --- Standardize and Validate Inputs ---

        # 1. Standardize `score_cols` to be a list
        if isinstance(score_cols, str):
            self.score_cols = [score_cols]
        elif isinstance(score_cols, list):
            self.score_cols = score_cols
        else:
            raise TypeError("'score_cols' must be a string or a list of strings.")

        # 2. Standardize `ascending` to be a list of booleans
        if isinstance(ascending, bool):
            self.ascending = [ascending] * len(self.score_cols)
        elif isinstance(ascending, list):
            self.ascending = ascending
        else:
            raise TypeError("'ascending' must be a boolean or a list of booleans.")

        # 3. Validate that the lengths match
        if len(self.score_cols) != len(self.ascending):
            raise ValueError(
                f"The number of score columns ({len(self.score_cols)}) must match "
                f"the number of ascending flags ({len(self.ascending)})."
            )

        if approximate and num_partitions <= 0:
            raise ValueError(
                "'num_partitions' must be positive for approximate ordering."
            )

        self.use_random_tiebreaker = use_random_tiebreaker
        self.approximate = approximate
        self.num_partitions = num_partitions
        self.folding = folding

    def _build_sort_expressions(self) -> List:
        """Helper method to construct the list of Spark sorting expressions."""
        sort_exprs = []

        # 1. Prepend folding column if enabled
        if self.folding > 1:
            sort_exprs.append(F.col("__fold_id__").asc())

        # 2. Add the primary score columns with specified directions
        for col_name, is_asc in zip(self.score_cols, self.ascending):
            if is_asc:
                sort_exprs.append(F.col(col_name).asc())
            else:
                sort_exprs.append(F.col(col_name).desc())

        # 3. Append random tie-breaker if enabled (for exact ordering)
        if not self.approximate and self.use_random_tiebreaker:
            sort_exprs.append(F.col("__random_tiebreaker__").asc())

        return sort_exprs

    def reorder(self, df: DataFrame) -> DataFrame:
        """Applies the reordering logic to the DataFrame."""

        temp_cols_to_drop = []

        # Add folding column if needed
        if self.folding > 1:
            print(f"Applying folding with {self.folding} folds.")
            df = df.withColumn(
                "__fold_id__",
                F.pmod(F.monotonically_increasing_id(), F.lit(self.folding)),
            )
            temp_cols_to_drop.append("__fold_id__")

        # Add random tie-breaker column if needed for exact sort
        if not self.approximate and self.use_random_tiebreaker:
            print("Enabling random tie-breaker to prevent data skew.")
            df = df.withColumn("__random_tiebreaker__", F.rand())
            temp_cols_to_drop.append("__random_tiebreaker__")

        # Build the final list of sorting expressions
        sort_expressions = self._build_sort_expressions()

        print(f"Sorting by: {sort_expressions}")

        # Apply the appropriate sort method
        if self.approximate:
            print(f"Using approximate ordering with {self.num_partitions} partitions.")
            sorted_df = df.repartition(self.num_partitions).sortWithinPartitions(
                *sort_expressions
            )
        else:
            print("Using exact ordering.")
            sorted_df = df.orderBy(*sort_expressions)

        # Clean up temporary columns
        if temp_cols_to_drop:
            return sorted_df.drop(*temp_cols_to_drop)
        else:
            return sorted_df

    def run(self) -> DataFrame:
        """Executes the pipeline node to fetch data and reorder it."""
        df = self.children[0].run()
        if len(self.children) > 1:
            for child in self.children[1:]:
                df = df.union(child.run())
        return self.reorder(df)


if __name__ == "__main__":
    spark = (
        SparkSession.builder.appName("ReorderNodeTest").master("local[*]").getOrCreate()
    )

    # Create a sample DataFrame for testing.
    # It has ties in both 'category' and 'score' to test multi-level sorting.
    data = [
        ("A", 100, "apple"),
        ("B", 200, "banana"),
        ("A", 100, "apricot"),
        ("C", 150, "cherry"),
        ("B", 300, "blueberry"),
        ("A", 200, "avocado"),
        ("B", 200, "blackberry"),
    ]
    schema = ["category", "score", "item"]
    df = spark.createDataFrame(data, schema)

    print("--- Original DataFrame ---")
    df.show()

    # --- Test Case 1: Simple single-column sort (descending) ---
    print("\n--- Test 1: Sort by 'score' descending ---")
    reorder_node_1 = Reorder(spark, score_cols="score", ascending=False)
    # reorder_node_1.children = [MockInputNode(spark, df)] # Inject DataFrame
    result_1 = reorder_node_1.reorder(df)
    result_1.show()

    # --- Test Case 2: Multi-column sort with mixed directions ---
    print("\n--- Test 2: Sort by 'category' ASC, then 'score' DESC ---")
    reorder_node_2 = Reorder(
        spark, score_cols=["category", "score"], ascending=[True, False]
    )
    # reorder_node_2.children = [MockInputNode(spark, df)]
    result_2 = reorder_node_2.reorder(df)
    result_2.show()

    # --- Test Case 3: Using the random tie-breaker ---
    # Sort by 'category' asc, 'score' asc. Notice that ('A', 100) is a tie.
    # The tie-breaker will ensure 'apple' and 'apricot' have a consistent,
    # though random, order across runs.
    print("\n--- Test 3: Sort by 'category' ASC, 'score' ASC with a tie-breaker ---")
    reorder_node_3 = Reorder(
        spark,
        score_cols=["category", "score"],
        ascending=True,
        use_random_tiebreaker=True,
    )
    # reorder_node_3.children = [MockInputNode(spark, df)]
    result_3 = reorder_node_3.reorder(df)
    result_3.show()

    # --- Test Case 4: Approximate sort within partitions ---
    print("\n--- Test 4: Approximate sort by 'item' descending ---")
    # The global order is not guaranteed, but data within each partition is sorted.
    reorder_node_4 = Reorder(
        spark, score_cols="item", ascending=False, approximate=True, num_partitions=2
    )
    result_4 = reorder_node_4.reorder(df)
    result_4.show()

    # --- Test Case 5: Using folding ---
    print("\n--- Test 5: Sort by 'score' ascending with 2 folds ---")
    # This will sort all rows with fold_id=0 first, then all rows with fold_id=1.
    reorder_node_5 = Reorder(spark, score_cols="score", ascending=True, folding=3)
    result_5 = reorder_node_5.reorder(df)
    result_5.show()

    spark.stop()
