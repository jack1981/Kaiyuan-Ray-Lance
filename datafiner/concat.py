"""Column concatenation node for row-wise string assembly.

This node concatenates configured columns into a single output column using a
pandas batch transform.
"""

from datafiner.base import PipelineNode
from datafiner.dataset_utils import map_batches_tuned, union_children
from datafiner.register import register


@register("Concat")
class Concat(PipelineNode):
    """
    Concatenate multiple input columns into one output column.

    Inputs/outputs:
        Reads child dataset(s) and returns dataset with `output_col` added.

    Side effects:
        Executes pandas `map_batches` transform.

    Assumptions:
        Input columns exist and values are string-convertible.
    """

    def __init__(
        self,
        runtime,
        select_cols: list,
        output_col: str,
        child_configs: list = None,
    ):
        """Configure source/output columns for concatenation.

        Args:
            runtime: Shared runtime config.
            select_cols: Ordered list of columns to concatenate.
            output_col: Target column for concatenated string result.
            child_configs: Upstream node configs.

        Returns:
            None.

        Side effects:
            None.

        Assumptions:
            `select_cols` is non-empty.
        """
        super().__init__(runtime, child_configs)
        if not isinstance(select_cols, list) or not select_cols:
            raise ValueError("'select_cols' must be a non-empty list.")
        self.select_cols = select_cols
        self.output_col = output_col

    def run(self):
        """Run row-wise concatenation over child dataset output.

        Inputs/outputs:
            Reads child dataset(s) and returns transformed dataset.

        Side effects:
            Prints operation info and schedules Ray batch transform.

        Assumptions:
            Nulls are treated as empty strings.
        """
        ds = union_children(self.children, by_name=False)

        print(f"[ColumnSelect] Selecting columns: {self.select_cols}")

        def concat_columns(batch):
            """Concatenate selected columns into one string column per row.

            Args:
                batch: Source pandas batch.

            Returns:
                Batch with `output_col` values.

            Side effects:
                None.

            Assumptions:
                Concatenation preserves input column order.
            """
            out = batch.copy()
            out[self.output_col] = out[self.select_cols].fillna("").astype(str).agg(
                "".join, axis=1
            )
            return out

        return map_batches_tuned(
            ds, self.runtime, concat_columns, batch_format="pandas"
        )
