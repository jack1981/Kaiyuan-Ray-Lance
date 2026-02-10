from datafiner.base import PipelineNode
from datafiner.dataset_utils import union_children
from datafiner.register import register


@register("Concat")
class Concat(PipelineNode):
    """
    A pipeline node to select a specific set of columns,
    dropping all others.
    """

    def __init__(
        self,
        runtime,
        select_cols: list,
        output_col: str,
        child_configs: list = None,
    ):
        """
        Initializes the ColumnSelect node.

        Args:
            runtime: Ray runtime configuration.
            select_cols (list): A list of column names to keep.
            child_configs (list, optional): List of child node configurations.
        """
        super().__init__(runtime, child_configs)
        if not isinstance(select_cols, list) or not select_cols:
            raise ValueError("'select_cols' must be a non-empty list.")
        self.select_cols = select_cols
        self.output_col = output_col

    def run(self):
        ds = union_children(self.children, by_name=False)

        print(f"[ColumnSelect] Selecting columns: {self.select_cols}")

        def concat_columns(batch):
            out = batch.copy()
            out[self.output_col] = out[self.select_cols].fillna("").astype(str).agg(
                "".join, axis=1
            )
            return out

        return ds.map_batches(concat_columns, batch_format="pandas")
