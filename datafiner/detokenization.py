import pandas as pd
from transformers import AutoTokenizer

from datafiner.base import PipelineNode
from datafiner.dataset_utils import map_batches_tuned, union_children
from datafiner.register import register


@register("Detokenization")
class Detokenization(PipelineNode):
    """
    Convert token IDs back to text.
    """

    def __init__(
        self,
        runtime,
        tokenizer_name_or_path: str,
        input_col: str = "text_tokenized",
        output_col: str = "text_decoded",
        skip_special_tokens: bool = True,
        clean_up_tokenization_spaces: bool = True,
        batch_size: int | None = None,
        concurrency: int | None = None,
        child_configs: list = None,
    ) -> None:
        super().__init__(runtime, child_configs)
        self.tokenizer_name_or_path = tokenizer_name_or_path
        self.input_col = input_col
        self.output_col = output_col
        self.skip_special_tokens = skip_special_tokens
        self.clean_up_tokenization_spaces = clean_up_tokenization_spaces
        self.batch_size = batch_size
        self.concurrency = concurrency

    def run(self):
        ds = union_children(self.children, by_name=False)
        return self.detokenize(ds)

    def detokenize(self, ds):
        tokenizer_name_or_path = self.tokenizer_name_or_path
        skip_special_tokens = self.skip_special_tokens
        clean_up_tokenization_spaces = self.clean_up_tokenization_spaces

        def decode_batch(batch: pd.DataFrame) -> pd.DataFrame:
            if not hasattr(decode_batch, "tokenizer"):
                decode_batch.tokenizer = AutoTokenizer.from_pretrained(
                    tokenizer_name_or_path
                )

            out = batch.copy()
            decoded = [None] * len(out)
            valid_indices = []
            valid_tokens = []
            for idx, token_ids in enumerate(out[self.input_col].tolist()):
                if isinstance(token_ids, (list, tuple)):
                    valid_indices.append(idx)
                    valid_tokens.append(list(token_ids))

            if valid_tokens:
                decoded_values = decode_batch.tokenizer.batch_decode(
                    valid_tokens,
                    skip_special_tokens=skip_special_tokens,
                    clean_up_tokenization_spaces=clean_up_tokenization_spaces,
                )
                for idx, text in zip(valid_indices, decoded_values):
                    decoded[idx] = text

            out[self.output_col] = decoded
            return out

        return map_batches_tuned(
            ds,
            self.runtime,
            decode_batch,
            batch_format="pandas",
            batch_size=self.batch_size,
            concurrency=self.concurrency,
        )


@register("BatchDetokenization")
class BatchDetokenization(Detokenization):
    """
    Batch detokenization alias. Ray map_batches already processes data in batches.
    """

    def __init__(
        self,
        runtime,
        tokenizer_name_or_path: str,
        input_col: str = "text_tokenized",
        output_col: str = "text_decoded",
        skip_special_tokens: bool = True,
        clean_up_tokenization_spaces: bool = True,
        batch_size: int = 1000,
        concurrency: int | None = None,
        child_configs: list = None,
    ) -> None:
        super().__init__(
            runtime=runtime,
            tokenizer_name_or_path=tokenizer_name_or_path,
            input_col=input_col,
            output_col=output_col,
            skip_special_tokens=skip_special_tokens,
            clean_up_tokenization_spaces=clean_up_tokenization_spaces,
            batch_size=batch_size,
            concurrency=concurrency,
            child_configs=child_configs,
        )
        self.batch_size = batch_size
