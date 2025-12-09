from pyspark.sql import SparkSession, DataFrame
from pyspark.sql import functions as F
from datafiner.base import PipelineNode
from datafiner.register import register
import pyspark.sql.functions as F
from abc import ABC, abstractmethod
from pyspark.sql.functions import col, concat_ws, transform


@register("AddConstants")
class AddConstants(PipelineNode):
    def __init__(
        self,
        spark: SparkSession,
        constant_list: list,
        column_list: list,
        child_configs=None,
    ):
        super().__init__(spark, child_configs)
        self.constant_list = constant_list
        self.column_list = column_list
        assert len(self.constant_list) == len(self.column_list), (
            f"columns and constants should match!"
        )

    def run(self):
        df = self.children[0].run()
        for column_name, constant in zip(self.column_list, self.constant_list):
            df = df.withColumn(column_name, F.lit(constant))
        return df


@register("ConversationToParagraph")
class ConversationToParagraph(PipelineNode):
    def __init__(
        self,
        spark: SparkSession,
        input_col: str,
        output_col: str,
        separator: str = "\n\n",
        field_key: str = "content",
        child_configs: list = None,
    ):
        super().__init__(spark, child_configs)
        self.input_col = input_col
        self.output_col = output_col
        self.separator = separator
        self.field_key = field_key

    def run(self):
        df = self.children[0].run()
        if len(self.children) > 1:
            for child in self.children[1:]:
                df = df.union(child.run())

        # Convert list of dicts (array<struct>) into single paragraph text
        return df.withColumn(
            self.output_col,
            concat_ws(
                self.separator,
                transform(col(self.input_col), lambda x: x.getField(self.field_key)),
            ),
        )


@register("ConcatenateColumns")
class ConcatenateColumns(PipelineNode):
    def __init__(
        self,
        spark: SparkSession,
        input_cols: list,
        output_col: str,
        separator: str = " ",
        child_configs: list = None,
    ):
        super().__init__(spark, child_configs)
        self.input_cols = input_cols
        self.output_col = output_col
        self.separator = separator

    def run(self):
        df = self.children[0].run()
        if len(self.children) > 1:
            for child in self.children[1:]:
                df = df.union(child.run())

        return df.withColumn(
            self.output_col,
            concat_ws(self.separator, *[col(c) for c in self.input_cols]),
        )
