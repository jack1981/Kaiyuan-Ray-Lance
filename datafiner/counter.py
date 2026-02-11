"""Token-count utility node for pre-tokenized sequence columns.

This node computes per-row token counts, optional duplicate-weighted totals, and
optional summary statistics for quick corpus diagnostics.
"""

import pandas as pd

from datafiner.base import PipelineNode
from datafiner.dataset_utils import map_batches_tuned, union_children
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
        """Configure token counting and summary behavior.

        Args:
            runtime: Shared runtime config.
            child_configs: Upstream node configs.
            input_col: Token sequence column (list/tuple expected).
            count_col: Output token-count column.
            with_duplicate_count: Whether to multiply by duplicate-count column.
            duplicate_count_col: Duplicate multiplier column name.
            summary: Whether to print aggregate totals.
            drop_intermediate: Whether to remove count column before returning.
            task_name: Optional label printed in summary output.

        Returns:
            None.

        Side effects:
            None during initialization.

        Assumptions:
            Non-sequence values in `input_col` contribute zero tokens.
        """
        super().__init__(runtime, child_configs)
        self.input_col = input_col
        self.with_duplicate_count = with_duplicate_count
        self.duplicate_count_col = duplicate_count_col
        self.summary = summary
        self.drop_intermediate = drop_intermediate
        self.count_col = count_col
        self.task_name = task_name

    def run(self):
        """Count tokens per row and optionally print aggregate summary metrics.

        Inputs/outputs:
            Reads child dataset(s) and returns counted dataset or dropped-count
            dataset depending on `drop_intermediate`.

        Side effects:
            Executes batch transforms, may materialize to pandas for summary, and
            prints aggregate statistics.

        Assumptions:
            Summary metrics are diagnostic and can incur full-dataset cost.
        """
        ds = union_children(self.children, by_name=False)

        def count_tokens(batch: pd.DataFrame) -> pd.DataFrame:
            """Compute token counts for one pandas batch.

            Args:
                batch: Source pandas batch.

            Returns:
                Batch with `count_col` populated.

            Side effects:
                None.

            Assumptions:
                Duplicate weighting multiplies numeric-cast values with nulls as 0.
            """
            out = batch.copy()
            out[self.count_col] = out[self.input_col].apply(
                lambda x: len(x) if isinstance(x, (list, tuple)) else 0
            )
            if self.with_duplicate_count:
                out[self.count_col] = (
                    pd.to_numeric(out[self.duplicate_count_col], errors="coerce").fillna(0)
                    * pd.to_numeric(out[self.count_col], errors="coerce").fillna(0)
                )
            return out

        counted = map_batches_tuned(
            ds, self.runtime, count_tokens, batch_format="pandas"
        )

        if self.summary:
            # NOTE(readability): Aggregate totals currently require pandas
            # materialization because this diagnostic path is not performance
            # critical in current usage.
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
            return map_batches_tuned(
                counted,
                self.runtime,
                lambda batch: batch.drop(columns=[self.count_col], errors="ignore"),
                batch_format="pandas",
            )
        return counted
