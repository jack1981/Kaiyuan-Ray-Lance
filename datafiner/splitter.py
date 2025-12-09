import yaml
import argparse
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql import functions as F
from datafiner.base import PipelineNode, PipelineTree
from datafiner.register import register
from typing import Tuple


@register("Splitter")
class Splitter(PipelineNode):
    """
    Splits a DataFrame into two disjoint sets (training and validation)
    and writes them to specified Parquet output paths.

    Includes two methods for splitting:
    - 'exact': Guarantees an exact number of rows in the training set.
    - 'fast_approximate': A faster, fully parallel method that results in an
      approximate number of training rows. Ideal for very large datasets.
    """

    def __init__(
        self,
        spark: SparkSession,
        train_file: str,
        val_file: str,
        num_train: int,
        split_method: str = "exact",
        shuffle: bool = False,
        num_train_files: int = None,
        num_val_files: int = None,
        mode: str = "overwrite",
        select_cols: list = None,
        shuffle_train: bool = False,
        child_configs: list = None,
    ) -> None:
        super().__init__(spark, child_configs)

        if not isinstance(num_train, int) or num_train <= 0:
            raise ValueError("'num_train' must be a positive integer.")
        if split_method not in ["exact", "fast_approximate"]:
            raise ValueError(
                "'split_method' must be either 'exact' or 'fast_approximate'."
            )

        self.train_file = train_file
        self.val_file = val_file
        self.num_train = num_train
        self.split_method = split_method
        self.shuffle = shuffle
        self.num_train_files = num_train_files
        self.num_val_files = num_val_files
        self.mode = mode
        self.select_cols = select_cols
        self.shuffle_train = shuffle_train

    def run(self) -> DataFrame:
        """Orchestrates the splitting and writing logic."""
        # 1. Prepare initial DataFrame
        df = self.children[0].run()
        if len(self.children) > 1:
            for child in self.children[1:]:
                df = df.union(child.run())

        if self.shuffle and self.split_method == "exact":
            print(
                "Shuffle is enabled for 'exact' method. Shuffling the entire dataset..."
            )
            df = df.orderBy(F.rand())

        if self.select_cols is not None:
            df = df.select(self.select_cols)

        # 2. Perform the split using the chosen method
        if self.split_method == "exact":
            train_df, val_df, df_to_unpersist = self._split_exact(df)
        else:  # fast_approximate
            train_df, val_df = self._split_fast_approximate(df)
            df_to_unpersist = None  # Fast method caches splits directly

        # 3. Write splits to disk with clear job descriptions
        self._write_split(
            train_df,
            self.train_file,
            self.num_train_files,
            self.shuffle_train,
            f"Splitter: Writing TRAIN split to {self.train_file}",
        )
        self._write_split(
            val_df,
            self.val_file,
            self.num_val_files,
            f"Splitter: Writing VALIDATION split to {self.val_file}",
        )

        # 4. Clean up any cached DataFrames
        if df_to_unpersist:
            df_to_unpersist.unpersist()
        if self.split_method == "fast_approximate":
            train_df.unpersist()
            val_df.unpersist()

        print("Splitting and writing completed successfully.")
        return val_df

    def _write_split(
        self, df: DataFrame, path: str, num_files: int, shuffle: bool, description: str
    ):
        """Helper function to write a DataFrame split with a job description."""
        print(description)
        # Set a clear description for this Spark job
        self.spark.sparkContext.setJobDescription(description)

        if shuffle:
            print("Shuffling training data before writing...")
            df = df.orderBy(F.rand())
            df.show(10)
        writer = df.write.mode(self.mode)
        if num_files:
            df.repartition(num_files).write.mode(self.mode).parquet(path)
        else:
            writer.parquet(path)

        # Clear the description after the action is complete
        self.spark.sparkContext.setJobDescription(None)

    def _split_exact(self, df: DataFrame) -> Tuple[DataFrame]:
        """Splits the DataFrame into an exact number of training rows."""
        print(f"Using 'exact' split method for {self.num_train} training rows.")
        tmp_id_col = "__temp_splitter_id__"
        df_with_id = df.withColumn(tmp_id_col, F.monotonically_increasing_id())

        # Cache the parent DataFrame to ensure consistency and avoid re-computation
        df_with_id.cache()

        train_df = df_with_id.limit(self.num_train)
        val_df = df_with_id.join(
            train_df.select(tmp_id_col), on=tmp_id_col, how="left_anti"
        )

        # Return the final splits and the temporary DF that needs to be unpersisted later
        return train_df.drop(tmp_id_col), val_df.drop(tmp_id_col), df_with_id

    def _split_fast_approximate(self, df: DataFrame) -> Tuple[DataFrame]:
        """Splits the DataFrame using a highly parallel, approximate method."""
        print(
            f"Using 'fast_approximate' split method for roughly {self.num_train} training rows."
        )

        # This action is necessary to calculate the ratio for randomSplit.
        count_desc = "Splitter (Fast): Calculating total row count for fraction"
        self.spark.sparkContext.setJobDescription(count_desc)
        total_count = df.count()
        self.spark.sparkContext.setJobDescription(None)

        if total_count == 0:
            return df, df  # Return empty DFs

        if self.num_train >= total_count:
            print(
                "Warning: 'num_train' is >= total rows. All data will be in the training set."
            )
            return df, self.spark.createDataFrame([], df.schema)

        fraction = self.num_train / total_count

        print(
            f"Total rows: {total_count}, Target train rows: {self.num_train}, Fraction: {fraction:.6f}"
        )
        df.show(10)

        # randomSplit is highly optimized and fully parallel.
        train_df, val_df = df.randomSplit(
            [fraction, 1.0 - fraction], seed=42 if self.shuffle else None
        )

        print("Post-split training DataFrame preview:")
        train_df.show(10)

        print("Post-split validation DataFrame preview:")
        val_df.show(10)

        # Cache the results of the split because we will perform two actions on them:
        # one for logging the count (optional) and one for writing the data.
        train_df.cache()
        val_df.cache()

        # Optional: Log the actual count for the training set. This is now a fast operation due to caching.
        log_count_desc = "Splitter (Fast): Logging actual train count"
        self.spark.sparkContext.setJobDescription(log_count_desc)
        actual_train_count = train_df.count()
        self.spark.sparkContext.setJobDescription(None)
        print(
            f"Approximation resulted in {actual_train_count} training rows (target was {self.num_train})."
        )

        return train_df, val_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=None)

    args = parser.parse_args()
    config = yaml.load(open(args.config, "r"), Loader=yaml.FullLoader)

    pipeline = PipelineTree(config)
    df = pipeline.run()
    df.show()
    print(df.count())
