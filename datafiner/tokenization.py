import pandas as pd
from transformers import AutoTokenizer

from datafiner.base import PipelineNode
from datafiner.dataset_utils import union_children
from datafiner.register import register


@register("Tokenization")
class Tokenization(PipelineNode):
    def __init__(
        self,
        runtime,
        tokenizer_name_or_path: str,
        input_col: str = "text",
        output_col: str = "text_tokenized",
        child_configs: list = None,
    ) -> None:
        super().__init__(runtime, child_configs)
        self.tokenizer_name_or_path = tokenizer_name_or_path
        self.input_col = input_col
        self.output_col = output_col

    def run(self):
        ds = union_children(self.children, by_name=False)
        return self.tokenize(ds)

    def tokenize(self, ds):
        tokenizer_name_or_path = self.tokenizer_name_or_path

        def encode_batch(batch: pd.DataFrame) -> pd.DataFrame:
            if not hasattr(encode_batch, "tokenizer"):
                encode_batch.tokenizer = AutoTokenizer.from_pretrained(
                    tokenizer_name_or_path
                )

            out = batch.copy()
            out[self.output_col] = out[self.input_col].fillna("").map(
                lambda x: encode_batch.tokenizer.encode(str(x))
            )
            return out

        return ds.map_batches(encode_batch, batch_format="pandas")
