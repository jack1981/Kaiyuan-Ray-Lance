import pandas as pd
from transformers import AutoTokenizer

from datafiner.base import PipelineNode
from datafiner.dataset_utils import map_batches_tuned, union_children
from datafiner.register import register


@register("Tokenization")
class Tokenization(PipelineNode):
    def __init__(
        self,
        runtime,
        tokenizer_name_or_path: str,
        input_col: str = "text",
        output_col: str = "text_tokenized",
        batch_size: int | None = None,
        concurrency: int | None = None,
        child_configs: list = None,
    ) -> None:
        super().__init__(runtime, child_configs)
        self.tokenizer_name_or_path = tokenizer_name_or_path
        self.input_col = input_col
        self.output_col = output_col
        self.batch_size = batch_size
        self.concurrency = concurrency

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
            texts = out[self.input_col].fillna("").astype(str).tolist()
            encoded = encode_batch.tokenizer(texts, add_special_tokens=True)
            out[self.output_col] = encoded["input_ids"]
            return out

        return map_batches_tuned(
            ds,
            self.runtime,
            encode_batch,
            batch_format="pandas",
            batch_size=self.batch_size,
            concurrency=self.concurrency,
        )
