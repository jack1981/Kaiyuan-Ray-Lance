from datafiner.base import PipelineNode
from datafiner.register import register
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql import functions as F
from pyspark.sql.types import FloatType, LongType


@register("DuplicateSampleRatio")
class DuplicateSampleRatio(PipelineNode):
    def __init__(
        self,
        spark: SparkSession,
        child_configs: list = None,
        global_sample_rate: float = 0.1,
        max_sample: float = 20.0,
        col: str = "duplicate_count",
    ):
        super().__init__(spark, child_configs)
        self.global_sample_rate = global_sample_rate
        self.max_sample = max_sample
        self.col = col

    def run(self):
        df = self.children[0].run()
        if len(self.children) > 1:
            for child in self.children[1:]:
                df = df.union(child.run())

        global_sample_rate = self.global_sample_rate
        max_sample = self.max_sample

        def get_sample_udf(duplicate_count):
            sample = duplicate_count * global_sample_rate
            if sample > max_sample:
                sample = max_sample
            return sample

        func = F.udf(get_sample_udf, FloatType())
        return df.withColumn(self.col, func(F.col(self.col)))


@register("Sampler")
class Sampler(PipelineNode):
    def __init__(
        self,
        spark: SparkSession,
        child_configs: list = None,
        col: str = "duplicate_count",
    ):
        super().__init__(spark, child_configs)
        self.col = col

    def run(self):
        df = self.children[0].run()
        if len(self.children) > 1:
            for child in self.children[1:]:
                df = df.union(child.run())
        return self.sample(df)

    def sample(self, df: DataFrame) -> DataFrame:
        # Convert the sampling column to integer type after applying probabilistic sampling
        # For float values like 2.7, we use floor + random to decide whether to round up
        df_with_counts = df.withColumn(
            self.col,  # Use a new column name to avoid confusion
            (
                F.floor(F.col(self.col))
                + F.when(
                    F.rand() < (F.col(self.col) - F.floor(F.col(self.col))), 1
                ).otherwise(0)
            ),
        )

        # Filter out rows where sample_count is 0
        df_with_counts = df_with_counts.filter(F.col(self.col) > 0)

        return df_with_counts


@register("Flatten")
class Flatten(PipelineNode):
    def __init__(
        self,
        spark: SparkSession,
        child_configs: list = None,
        col: str = "duplicate_count",
    ):
        super().__init__(spark, child_configs)
        self.col = col

    def run(self):
        df = self.children[0].run()
        if len(self.children) > 1:
            for child in self.children[1:]:
                df = df.union(child.run())
        return self.flatten(df)

    def flatten(self, df: DataFrame) -> DataFrame:
        # Ensure the column is integer type for sequence generation
        # df = df.withColumn(self.col, F.col(self.col).cast(LongType()))

        # explode the data
        df_with_array = df.withColumn(
            "repeat_array", F.expr(f"sequence(1, {self.col})")
        )
        df_with_array.show()

        # expand the array, generate repeated rows
        df_expanded = df_with_array.select(
            *[col for col in df.columns if col != self.col],
            F.explode("repeat_array").alias("_dummy"),
        ).drop(
            "_dummy", "repeat_array"
        )  # Remove repeat_count reference since it doesn't exist

        return df_expanded


@register("GroupFlatten")
class GroupFlatten(PipelineNode):
    def __init__(
        self,
        spark: SparkSession,
        child_configs: list = None,
        cols: list = None,
        sub_cols: list = None,
        output_cols: list = None,
    ):
        super().__init__(spark, child_configs)
        assert cols is not None and len(cols) >= 1, (
            "GroupFlatten requires at least one array column."
        )
        self.cols = cols
        self.sub_cols = sub_cols if sub_cols is not None else cols
        self.output_cols = output_cols if output_cols is not None else self.sub_cols
        assert len(self.sub_cols) == len(self.output_cols), (
            "sub_cols and output_cols must have the same length."
        )

    def run(self):
        df = self.children[0].run()
        if len(self.children) > 1:
            for child in self.children[1:]:
                df = df.union(child.run())
        return self.flatten(df)

    def flatten(self, df: DataFrame) -> DataFrame:
        # zip all arrays (array<struct>)
        df = df.withColumn("zipped", F.arrays_zip(*self.cols))

        # explode the zipped array
        df = df.withColumn("z", F.explode("zipped")).drop("zipped")

        for idx, c in enumerate(self.sub_cols):
            c_out = self.output_cols[idx]
            df = df.withColumn(c_out, F.col(f"z.{c}"))  # get map value

        df = df.drop("z")
        df.show()
        return df
