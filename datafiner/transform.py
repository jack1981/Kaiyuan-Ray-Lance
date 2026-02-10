import pandas as pd

from datafiner.base import PipelineNode
from datafiner.dataset_utils import union_children
from datafiner.register import register


@register("AddConstants")
class AddConstants(PipelineNode):
    def __init__(
        self,
        runtime,
        constant_list: list,
        column_list: list,
        child_configs=None,
    ):
        super().__init__(runtime, child_configs)
        self.constant_list = constant_list
        self.column_list = column_list
        assert len(self.constant_list) == len(self.column_list), (
            "columns and constants should match!"
        )

    def run(self):
        ds = union_children(self.children, by_name=False)

        def add_constants(batch: pd.DataFrame) -> pd.DataFrame:
            out = batch.copy()
            for column_name, constant in zip(self.column_list, self.constant_list):
                out[column_name] = constant
            return out

        return ds.map_batches(add_constants, batch_format="pandas")


@register("ConversationToParagraph")
class ConversationToParagraph(PipelineNode):
    def __init__(
        self,
        runtime,
        input_col: str,
        output_col: str,
        separator: str = "\n\n",
        field_key: str = "content",
        child_configs: list = None,
    ):
        super().__init__(runtime, child_configs)
        self.input_col = input_col
        self.output_col = output_col
        self.separator = separator
        self.field_key = field_key

    def run(self):
        ds = union_children(self.children, by_name=False)

        def convert(batch: pd.DataFrame) -> pd.DataFrame:
            out = batch.copy()

            def to_paragraph(value):
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

        return ds.map_batches(convert, batch_format="pandas")


@register("ConcatenateColumns")
class ConcatenateColumns(PipelineNode):
    def __init__(
        self,
        runtime,
        input_cols: list,
        output_col: str,
        separator: str = " ",
        child_configs: list = None,
    ):
        super().__init__(runtime, child_configs)
        self.input_cols = input_cols
        self.output_col = output_col
        self.separator = separator

    def run(self):
        ds = union_children(self.children, by_name=False)

        def concat(batch: pd.DataFrame) -> pd.DataFrame:
            out = batch.copy()
            out[self.output_col] = (
                out[self.input_cols]
                .fillna("")
                .astype(str)
                .agg(self.separator.join, axis=1)
                .str.strip()
            )
            return out

        return ds.map_batches(concat, batch_format="pandas")
