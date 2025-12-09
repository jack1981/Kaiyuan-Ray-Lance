from pyspark.sql import SparkSession, DataFrame
from pyspark.sql import functions as F
from pyspark.sql.types import StringType, ArrayType, IntegerType
from transformers import AutoTokenizer
from datafiner.base import PipelineNode
from datafiner.register import register


@register("Detokenization")
class Detokenization(PipelineNode):
    """
    Detokenization pipeline node that converts token IDs back to text.

    Args:
        spark: SparkSession instance
        tokenizer_name_or_path: Path or name of the pretrained tokenizer
        input_col: Column name containing token IDs (default: "text_tokenized")
        output_col: Column name for decoded text output (default: "text_decoded")
        skip_special_tokens: Whether to skip special tokens in decoding (default: True)
        clean_up_tokenization_spaces: Whether to clean up extra spaces (default: True)
        child_configs: List of child pipeline configurations

    Example:
        ```python
        detokenizer = Detokenization(
            spark=spark,
            tokenizer_name_or_path="bert-base-uncased",
            input_col="text_tokenized",
            output_col="text_decoded",
            skip_special_tokens=True
        )
        df_decoded = detokenizer.run()
        ```
    """

    def __init__(
        self,
        spark: SparkSession,
        tokenizer_name_or_path: str,
        input_col: str = "text_tokenized",
        output_col: str = "text_decoded",
        skip_special_tokens: bool = True,
        clean_up_tokenization_spaces: bool = True,
        child_configs: list = None,
    ) -> None:
        super().__init__(spark, child_configs)
        self.tokenizer_name_or_path = tokenizer_name_or_path
        self.input_col = input_col
        self.output_col = output_col
        self.skip_special_tokens = skip_special_tokens
        self.clean_up_tokenization_spaces = clean_up_tokenization_spaces

    def run(self) -> DataFrame:
        """
        Execute the detokenization pipeline.

        Returns:
            DataFrame with detokenized text column added
        """
        df = self.children[0].run()
        if len(self.children) > 1:
            for child in self.children[1:]:
                df = df.union(child.run())
        return self.detokenize(df)

    def detokenize(self, df: DataFrame) -> DataFrame:
        """
        Apply detokenization to the input DataFrame.

        Args:
            df: Input DataFrame containing token IDs column

        Returns:
            DataFrame with decoded text column added
        """
        tokenizer_name_or_path = self.tokenizer_name_or_path
        skip_special_tokens = self.skip_special_tokens
        clean_up_tokenization_spaces = self.clean_up_tokenization_spaces

        def decode_udf(token_ids):
            """
            UDF to decode token IDs back to text.
            Uses lazy loading to initialize tokenizer once per executor.
            """
            if token_ids is None:
                return None

            if not hasattr(decode_udf, "tokenizer"):
                decode_udf.tokenizer = AutoTokenizer.from_pretrained(
                    tokenizer_name_or_path
                )

            return decode_udf.tokenizer.decode(
                token_ids,
                skip_special_tokens=skip_special_tokens,
                clean_up_tokenization_spaces=clean_up_tokenization_spaces,
            )

        func = F.udf(decode_udf, StringType())
        return df.withColumn(self.output_col, func(F.col(self.input_col)))


# Optional: Batch detokenization for better performance
@register("BatchDetokenization")
class BatchDetokenization(PipelineNode):
    """
    Batch detokenization pipeline node for improved performance.
    Processes multiple rows at once using pandas UDF.

    Args:
        spark: SparkSession instance
        tokenizer_name_or_path: Path or name of the pretrained tokenizer
        input_col: Column name containing token IDs (default: "text_tokenized")
        output_col: Column name for decoded text output (default: "text_decoded")
        skip_special_tokens: Whether to skip special tokens in decoding (default: True)
        clean_up_tokenization_spaces: Whether to clean up extra spaces (default: True)
        batch_size: Number of rows to process in each batch (default: 1000)
        child_configs: List of child pipeline configurations

    Example:
        ```python
        batch_detokenizer = BatchDetokenization(
            spark=spark,
            tokenizer_name_or_path="gpt2",
            input_col="text_tokenized",
            output_col="text_decoded",
            batch_size=1000
        )
        df_decoded = batch_detokenizer.run()
        ```
    """

    def __init__(
        self,
        spark: SparkSession,
        tokenizer_name_or_path: str,
        input_col: str = "text_tokenized",
        output_col: str = "text_decoded",
        skip_special_tokens: bool = True,
        clean_up_tokenization_spaces: bool = True,
        batch_size: int = 1000,
        child_configs: list = None,
    ) -> None:
        super().__init__(spark, child_configs)
        self.tokenizer_name_or_path = tokenizer_name_or_path
        self.input_col = input_col
        self.output_col = output_col
        self.skip_special_tokens = skip_special_tokens
        self.clean_up_tokenization_spaces = clean_up_tokenization_spaces
        self.batch_size = batch_size

    def run(self) -> DataFrame:
        """
        Execute the batch detokenization pipeline.

        Returns:
            DataFrame with detokenized text column added
        """
        df = self.children[0].run()
        if len(self.children) > 1:
            for child in self.children[1:]:
                df = df.union(child.run())
        return self.detokenize(df)

    def detokenize(self, df: DataFrame) -> DataFrame:
        """
        Apply batch detokenization to the input DataFrame.

        Args:
            df: Input DataFrame containing token IDs column

        Returns:
            DataFrame with decoded text column added
        """
        from pyspark.sql.functions import pandas_udf
        import pandas as pd

        tokenizer_name_or_path = self.tokenizer_name_or_path
        skip_special_tokens = self.skip_special_tokens
        clean_up_tokenization_spaces = self.clean_up_tokenization_spaces

        @pandas_udf(StringType())
        def decode_batch_udf(token_ids_series: pd.Series) -> pd.Series:
            """
            Pandas UDF to decode batches of token IDs back to text.
            More efficient for large datasets.
            """
            if not hasattr(decode_batch_udf, "tokenizer"):
                decode_batch_udf.tokenizer = AutoTokenizer.from_pretrained(
                    tokenizer_name_or_path
                )

            tokenizer = decode_batch_udf.tokenizer

            # Handle None values
            results = []
            for token_ids in token_ids_series:
                if token_ids is None or len(token_ids) == 0:
                    results.append(None)
                else:
                    decoded = tokenizer.decode(
                        token_ids,
                        skip_special_tokens=skip_special_tokens,
                        clean_up_tokenization_spaces=clean_up_tokenization_spaces,
                    )
                    results.append(decoded)

            return pd.Series(results)

        return df.withColumn(self.output_col, decode_batch_udf(F.col(self.input_col)))
