"""Tokenizer-based transform node built on Hugging Face tokenizers.

This node lazily initializes tokenizer instances per worker batch function and
writes token id lists into a configured output column.
"""

import pandas as pd
from transformers import AutoTokenizer

from datafiner.base import PipelineNode
from datafiner.dataset_utils import map_batches_tuned, union_children
from datafiner.register import register


@register("Tokenization")
class Tokenization(PipelineNode):
    """Tokenize text columns into token-id sequences.

    Inputs/outputs:
        Reads child dataset(s) and writes token ids to `output_col`.

    Side effects:
        Loads tokenizer model files and executes distributed batch transforms.

    Assumptions:
        `input_col` values are string-convertible and tokenizer path is valid.
    """

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
        """Configure tokenizer path and batch execution controls.

        Args:
            runtime: Shared runtime config.
            tokenizer_name_or_path: HF model id or local tokenizer path.
            input_col: Source text column.
            output_col: Destination token-id column.
            batch_size: Optional per-stage batch size override.
            concurrency: Optional per-stage concurrency override.
            child_configs: Upstream node configs.

        Returns:
            None.

        Side effects:
            None during initialization.

        Assumptions:
            Tokenizer should be loaded lazily inside workers for scalability.
        """
        super().__init__(runtime, child_configs)
        self.tokenizer_name_or_path = tokenizer_name_or_path
        self.input_col = input_col
        self.output_col = output_col
        self.batch_size = batch_size
        self.concurrency = concurrency

    def run(self):
        """Execute tokenization over child dataset output.

        Inputs/outputs:
            Reads child dataset(s) and returns tokenized dataset.

        Side effects:
            Delegates to `tokenize`, which loads tokenizer and runs map batches.

        Assumptions:
            Child datasets are union-compatible by position.
        """
        ds = union_children(self.children, by_name=False)
        return self.tokenize(ds)

    def tokenize(self, ds):
        """Tokenize text values and append token-id lists.

        Args:
            ds: Source dataset containing text column.

        Returns:
            Tokenized dataset.

        Side effects:
            Loads tokenizer model once per worker and runs distributed map tasks.

        Assumptions:
            Empty/null text values are treated as empty strings.
        """
        tokenizer_name_or_path = self.tokenizer_name_or_path

        def encode_batch(batch: pd.DataFrame) -> pd.DataFrame:
            """Tokenize one pandas batch using lazily cached tokenizer.

            Args:
                batch: Source pandas batch.

            Returns:
                Batch with token ids in `output_col`.

            Side effects:
                Downloads/loads tokenizer assets when first invoked per worker.

            Assumptions:
                Worker-local function attributes persist across batch calls.
            """
            if not hasattr(encode_batch, "tokenizer"):
                # NOTE(readability): Lazy worker-local initialization avoids
                # repeatedly constructing tokenizers for each micro-batch.
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
