"""Batch-wise text/column transformation nodes built on Ray map_batches.

These transforms keep operations in pandas batches for predictable vectorized
behavior and avoid row-wise Python UDF overhead.
See also `datafiner/dataset_utils.py`.
"""

import pandas as pd

from datafiner.base import PipelineNode
from datafiner.dataset_utils import map_batches_tuned, union_children
from datafiner.register import register


@register("AddConstants")
class AddConstants(PipelineNode):
    """Add constant-valued columns to every record.

    Inputs/outputs:
        Reads child dataset(s) and returns dataset with added columns.

    Side effects:
        Executes pandas batch transforms in Ray tasks.

    Assumptions:
        `constant_list` and `column_list` must be aligned by position.
    """

    def __init__(
        self,
        runtime,
        constant_list: list,
        column_list: list,
        child_configs=None,
    ):
        """Configure constant-column assignment.

        Args:
            runtime: Shared runtime config.
            constant_list: Constant values to assign.
            column_list: Target column names.
            child_configs: Upstream node configs.

        Returns:
            None.

        Side effects:
            None.

        Assumptions:
            Lengths of constants and column names are identical.
        """
        super().__init__(runtime, child_configs)
        self.constant_list = constant_list
        self.column_list = column_list
        assert len(self.constant_list) == len(self.column_list), (
            "columns and constants should match!"
        )

    def run(self):
        """Apply constant assignment to each batch.

        Inputs/outputs:
            Reads child dataset(s) and returns transformed dataset.

        Side effects:
            Runs `map_batches` over all data.

        Assumptions:
            Children are union-compatible by position.
        """
        ds = union_children(self.children, by_name=False)

        def add_constants(batch: pd.DataFrame) -> pd.DataFrame:
            """Assign configured constant values to output columns.

            Args:
                batch: Source pandas batch.

            Returns:
                Batch with added/overwritten constant columns.

            Side effects:
                None.

            Assumptions:
                Constant assignment is idempotent for repeated runs.
            """
            out = batch.copy()
            for column_name, constant in zip(self.column_list, self.constant_list):
                out[column_name] = constant
            return out

        return map_batches_tuned(ds, self.runtime, add_constants, batch_format="pandas")


@register("ConversationToParagraph")
class ConversationToParagraph(PipelineNode):
    """Flatten conversation-like list structures into paragraph strings.

    Inputs/outputs:
        Reads list/dict conversation column and writes text to `output_col`.

    Side effects:
        Executes pandas batch transforms.

    Assumptions:
        Conversation entries may be dicts containing `field_key` or scalar values.
    """

    def __init__(
        self,
        runtime,
        input_col: str,
        output_col: str,
        separator: str = "\n\n",
        field_key: str = "content",
        child_configs: list = None,
    ):
        """Configure conversation flattening behavior.

        Args:
            runtime: Shared runtime config.
            input_col: Source conversation column.
            output_col: Destination paragraph column.
            separator: Join delimiter between conversation parts.
            field_key: Dict key used for dict-form messages.
            child_configs: Upstream node configs.

        Returns:
            None.

        Side effects:
            None.

        Assumptions:
            Non-list values are stringified directly.
        """
        super().__init__(runtime, child_configs)
        self.input_col = input_col
        self.output_col = output_col
        self.separator = separator
        self.field_key = field_key

    def run(self):
        """Convert conversation structures into paragraph text per row.

        Inputs/outputs:
            Reads child dataset(s) and returns transformed dataset.

        Side effects:
            Runs `map_batches` over all rows.

        Assumptions:
            Missing conversation values map to `None`.
        """
        ds = union_children(self.children, by_name=False)

        def convert(batch: pd.DataFrame) -> pd.DataFrame:
            """Convert each conversation value in a batch to paragraph text.

            Args:
                batch: Source pandas batch.

            Returns:
                Batch with `output_col` populated.

            Side effects:
                None.

            Assumptions:
                Item ordering inside conversations should be preserved.
            """
            out = batch.copy()

            def to_paragraph(value):
                """Convert one row value to normalized paragraph string.

                Args:
                    value: Conversation value (list/tuple/dict/scalar/None).

                Returns:
                    Joined paragraph string or `None`.

                Side effects:
                    None.

                Assumptions:
                    Dict items expose textual content under `field_key`.
                """
                if value is None:
                    return None
                if not isinstance(value, (list, tuple)):
                    return str(value)
                parts = []
                for item in value:
                    if isinstance(item, dict):
                        part = item.get(self.field_key)
                    else:
                        part = item
                    if part is not None:
                        parts.append(str(part))
                return self.separator.join(parts)

            out[self.output_col] = out[self.input_col].map(to_paragraph)
            return out

        return map_batches_tuned(ds, self.runtime, convert, batch_format="pandas")


@register("ConcatenateColumns")
class ConcatenateColumns(PipelineNode):
    """Concatenate multiple columns into one text column.

    Inputs/outputs:
        Reads configured input columns and writes combined value to output column.

    Side effects:
        Executes pandas batch transforms.

    Assumptions:
        Null inputs should be treated as empty strings before concatenation.
    """

    def __init__(
        self,
        runtime,
        input_cols: list,
        output_col: str,
        separator: str = " ",
        child_configs: list = None,
    ):
        """Configure multi-column concatenation.

        Args:
            runtime: Shared runtime config.
            input_cols: Ordered columns to combine.
            output_col: Destination combined column.
            separator: Delimiter between input values.
            child_configs: Upstream node configs.

        Returns:
            None.

        Side effects:
            None.

        Assumptions:
            Inputs are string-convertible after null fill.
        """
        super().__init__(runtime, child_configs)
        self.input_cols = input_cols
        self.output_col = output_col
        self.separator = separator

    def run(self):
        """Concatenate configured columns for every row.

        Inputs/outputs:
            Reads child dataset(s) and returns transformed dataset.

        Side effects:
            Runs a pandas batch transform.

        Assumptions:
            Column order in `input_cols` defines concatenation order.
        """
        ds = union_children(self.children, by_name=False)

        def concat(batch: pd.DataFrame) -> pd.DataFrame:
            """Create one combined string column from multiple source columns.

            Args:
                batch: Source pandas batch.

            Returns:
                Batch with `output_col` filled.

            Side effects:
                None.

            Assumptions:
                Whitespace trimming preserves human-readable output text.
            """
            out = batch.copy()
            out[self.output_col] = (
                out[self.input_cols]
                .fillna("")
                .astype(str)
                .agg(self.separator.join, axis=1)
                .str.strip()
            )
            return out

        return map_batches_tuned(ds, self.runtime, concat, batch_format="pandas")
