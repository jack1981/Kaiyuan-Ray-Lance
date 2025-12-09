from pyspark.sql import SparkSession, DataFrame
from pyspark.sql import functions as F
from pyspark.sql.types import ArrayType, IntegerType
from transformers import AutoTokenizer
from datafiner.base import PipelineNode
from datafiner.register import register


@register("Tokenization")
class Tokenization(PipelineNode):
    def __init__(
        self,
        spark: SparkSession,
        tokenizer_name_or_path: str,
        input_col: str = "text",
        output_col: str = "text_tokenized",
        child_configs: list = None,
    ) -> None:
        super().__init__(spark, child_configs)
        self.tokenizer_name_or_path = tokenizer_name_or_path
        self.input_col = input_col
        self.output_col = output_col

    def run(self):
        df = self.children[0].run()
        if len(self.children) > 1:
            for child in self.children[1:]:
                df = df.union(child.run())
        return self.tokenize(df)

    def tokenize(self, df: DataFrame) -> DataFrame:
        tokenizer_name_or_path = self.tokenizer_name_or_path

        def encode_udf(text):
            if not hasattr(encode_udf, "tokenizer"):
                encode_udf.tokenizer = AutoTokenizer.from_pretrained(
                    tokenizer_name_or_path
                )
            return encode_udf.tokenizer.encode(text)

        func = F.udf(encode_udf, ArrayType(IntegerType()))
        return df.withColumn(self.output_col, func(F.col(self.input_col)))
