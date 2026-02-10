import pandas as pd

from datafiner.base import PipelineNode
from datafiner.dataset_utils import union_children
from datafiner.register import register


@register("TokenCounter_v2")
class TokenCounter_v2(PipelineNode):
    """
    Count tokens in a pre-tokenized column.
    """

    def __init__(
        self,
        runtime,
        child_configs: list = None,
        input_col: str = "text_tokenized",
        count_col: str = "token_count",
        with_duplicate_count: bool = False,
        duplicate_count_col: str = "duplicate_count",
        summary: bool = True,
        drop_intermediate: bool = False,
        task_name: str = None,
    ):
        super().__init__(runtime, child_configs)
        self.input_col = input_col
        self.with_duplicate_count = with_duplicate_count
        self.duplicate_count_col = duplicate_count_col
        self.summary = summary
        self.drop_intermediate = drop_intermediate
        self.count_col = count_col
        self.task_name = task_name

    def run(self):
        ds = union_children(self.children, by_name=False)

        def count_tokens(batch: pd.DataFrame) -> pd.DataFrame:
            out = batch.copy()
            out[self.count_col] = out[self.input_col].map(
                lambda x: len(x) if isinstance(x, (list, tuple)) else 0
            )
            if self.with_duplicate_count:
                out[self.count_col] = (
                    pd.to_numeric(out[self.duplicate_count_col], errors="coerce").fillna(0)
                    * pd.to_numeric(out[self.count_col], errors="coerce").fillna(0)
                )
            return out

        counted = ds.map_batches(count_tokens, batch_format="pandas")

        if self.summary:
            pdf = counted.to_pandas()
            total_tokens = float(pdf[self.count_col].sum()) if not pdf.empty else 0.0
            if self.task_name:
                print(f"Task: {self.task_name}")
            print(f"Total token count: {total_tokens}")

            if self.with_duplicate_count and self.duplicate_count_col in pdf.columns:
                row_count = float(pdf[self.duplicate_count_col].sum())
            else:
                row_count = float(len(pdf))

            avg_tokens = total_tokens / row_count if row_count > 0 else 0
            print(f"Total rows (or documents): {row_count}")
            print(f"Average tokens per row: {avg_tokens:.2f}")

        if self.drop_intermediate:
            return counted.map_batches(
                lambda batch: batch.drop(columns=[self.count_col], errors="ignore"),
                batch_format="pandas",
            )
        return counted
