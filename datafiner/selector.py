# In Selector class
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql import functions as F
from datafiner.base import PipelineNode
from datafiner.register import register
from pyspark.sql.window import Window


@register("Selector")
class Selector(PipelineNode):
    """
    Selects a subset of rows from a DataFrame using various methods.

    Methods:
    - 'head': Selects the first n rows. Non-deterministic without an order.
    - 'tail': Selects n rows that are NOT in the 'head' set. Useful for creating a disjoint holdout set without a full shuffle.
    - 'random': Selects a true random sample of n rows. This requires a full shuffle and is expensive.
    - 'fast_random': Selects an approximate random sample of n rows using stratified sampling. Much faster but not exact.
    - 'approximate': An efficient method to select top n rows per partition based on a score column.
    """

    def __init__(
        self,
        spark: SparkSession,
        num_rows: int = None,
        selection_ratio: float = None,
        method: str = "random",
        score_col: str = None,
        approximate: bool = False,
        num_partitions: int = 32,
        seed: int = None,
        child_configs: list = None,
    ) -> None:
        super().__init__(spark, child_configs)

        # 'fast_random' is a new, more efficient method
        valid_methods = ["random", "head", "tail", "depr_fast_random", "fast_random"]

        if approximate:
            valid_methods.append("approximate")
            if not score_col:
                raise ValueError(
                    "When 'approximate' is True, 'score_col' must be provided."
                )
            if num_partitions <= 0:
                raise ValueError(
                    "'num_partitions' must be a positive integer when 'approximate' is True."
                )

        if method not in valid_methods:
            raise ValueError(
                f"Method '{method}' is not supported. Choose 'random', 'head', or 'fast_random'."
            )

        if num_rows is None and selection_ratio is None:
            raise ValueError("Either 'num_rows' or 'selection_ratio' must be provided.")

        if num_rows is not None and selection_ratio is not None:
            raise ValueError(
                "Only one of 'num_rows' or 'selection_ratio' should be provided, not both."
            )

        if num_rows is not None and (not isinstance(num_rows, int) or num_rows <= 0):
            raise ValueError("'num_rows' must be a positive integer.")

        if selection_ratio is not None and (
            not isinstance(selection_ratio, float) or not (0 < selection_ratio <= 1.0)
        ):
            raise ValueError("'selection_ratio' must be a float in the range (0, 1].")

        self.num_rows = num_rows
        self.selection_ratio = selection_ratio
        self.method = method
        self.score_col = score_col
        self.approximate = approximate
        self.num_partitions = num_partitions
        self.seed = seed

    def run(self) -> DataFrame:
        df = self.children[0].run()
        if len(self.children) > 1:
            for child in self.children[1:]:
                df = df.union(child.run())
        return self.select(df)

    def select(self, df: DataFrame) -> DataFrame:
        print(f"Sampling {self.num_rows} rows using '{self.method}' method.")

        if self.method == "head":
            return df.limit(self.num_rows)

        elif self.method == "tail":
            tmp_id_col = "__tmp_id__"
            df_with_id = df.withColumn(tmp_id_col, F.monotonically_increasing_id())
            df_with_id.cache()
            head_ids_df = df_with_id.limit(self.num_rows).select(tmp_id_col)
            ids_to_exclude = [row[tmp_id_col] for row in head_ids_df.collect()]
            tail_df = df_with_id.filter(~F.col(tmp_id_col).isin(ids_to_exclude)).drop(
                tmp_id_col
            )
            df_with_id.unpersist()
            return tail_df

        elif self.method == "random":
            # Exact but expensive
            if self.selection_ratio is not None:
                return df.sample(
                    withReplacement=False, fraction=self.selection_ratio, seed=self.seed
                )
            else:
                total_count = df.count()
                if total_count == 0:
                    return df
                fraction = self.num_rows / total_count
                return df.sample(
                    withReplacement=False, fraction=fraction, seed=self.seed
                )

        elif self.method == "depr_fast_random":
            # Approximate but much more efficient
            total_count = df.count()
            if total_count == 0:
                return df

            fraction = self.num_rows / total_count
            # Ensure fraction is not > 1.0, which can cause errors
            fraction = min(
                fraction * 1.1, 1.0
            )  # Add a small buffer to get closer to num_rows

            print(f"Total rows: {total_count}, sampling with fraction: {fraction}")
            return df.sample(
                withReplacement=False, fraction=fraction, seed=self.seed
            ).limit(self.num_rows)
        elif self.method == "fast_random":
            num_partitions = df.rdd.getNumPartitions()
            if num_partitions == 0:
                return df  # No partitioning, return as is
            elif num_partitions > self.num_rows:
                num_partitions = (
                    self.num_rows // 10000 + 1
                )  # Ensure at least 10k rows per partition
            print(f"Using {num_partitions} partitions for fast random sampling.")
            limit_per_partition = self.num_rows // num_partitions + 1
            window_spec = Window.partitionBy(F.spark_partition_id()).orderBy(
                F.rand(self.seed)
            )
            df_with_rank = df.withColumn("rank", F.row_number().over(window_spec))
            return df_with_rank.filter(F.col("rank") <= limit_per_partition).drop(
                "rank"
            )
        elif self.method == "approximate":
            repartitioned_df = df.repartition(self.num_partitions)
            window_spec = Window.partitionBy(F.spark_partition_id()).orderBy(
                F.col(self.score_col).desc()
            )
            df_with_rank = repartitioned_df.withColumn(
                "rank", F.row_number().over(window_spec)
            )
            if self.num_rows is None:
                total_count = df.count()
                self.num_rows = total_count * self.selection_ratio
            limit_per_partition = self.num_rows // self.num_partitions
            return df_with_rank.filter(F.col("rank") <= limit_per_partition).drop(
                "rank"
            )
