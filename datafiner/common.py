from datafiner.base import PipelineNode
from datafiner.dataset_utils import (
    drop_columns,
    map_batches_tuned,
    select_columns,
    show_dataset,
    union_children,
)
from datafiner.register import register


@register("ColumnSelect")
class ColumnSelect(PipelineNode):
    """
    Select a specific set of columns.
    """

    def __init__(
        self,
        runtime,
        select_cols: list,
        child_configs: list = None,
    ):
        super().__init__(runtime, child_configs)
        if not isinstance(select_cols, list) or not select_cols:
            raise ValueError("'select_cols' must be a non-empty list.")
        self.select_cols = select_cols

    def run(self):
        ds = union_children(self.children, by_name=False)
        print(f"[ColumnSelect] Selecting columns: {self.select_cols}")
        return select_columns(ds, self.select_cols, runtime=self.runtime)


@register("ColumnDrop")
class ColumnDrop(PipelineNode):
    """
    Drop a specific set of columns.
    """

    def __init__(
        self,
        runtime,
        drop_cols: list,
        child_configs: list = None,
    ):
        super().__init__(runtime, child_configs)
        if not isinstance(drop_cols, list) or not drop_cols:
            raise ValueError("'drop_cols' must be a non-empty list.")
        self.drop_cols = drop_cols

    def run(self):
        ds = union_children(self.children, by_name=False)
        print(f"[ColumnDrop] Dropping columns: {self.drop_cols}")
        return drop_columns(ds, self.drop_cols, runtime=self.runtime)


@register("ColumnAlias")
class ColumnAlias(PipelineNode):
    def __init__(
        self,
        runtime,
        input_col: str,
        output_col: str,
        child_configs: list = None,
    ):
        super().__init__(runtime, child_configs)
        self.input_col = input_col
        self.output_col = output_col

    def run(self):
        ds = union_children(self.children, by_name=False)

        def alias_batch(batch):
            out = batch.copy()
            if self.input_col in out.columns:
                out = out.rename(columns={self.input_col: self.output_col})
            return out

        return map_batches_tuned(ds, self.runtime, alias_batch, batch_format="pandas")


@register("Schema")
class Schema(PipelineNode):
    """
    Print schema.
    """

    def __init__(
        self,
        runtime,
        child_configs: list = None,
    ):
        super().__init__(runtime, child_configs)

    def run(self):
        ds = union_children(self.children, by_name=False)
        print("\n--- Dataset Schema ---")
        print(ds.schema())
        print("----------------------\n")
        return ds


@register("Row Number")
class RowNumber(PipelineNode):
    """
    Print row count.
    """

    def __init__(
        self,
        runtime,
        child_configs: list = None,
    ):
        super().__init__(runtime, child_configs)

    def run(self):
        ds = union_children(self.children, by_name=False)
        count = ds.count()
        print("\n--- Dataset Row Count ---")
        print(f"Total Rows: {count}")
        print("-------------------------\n")
        return ds


@register("Stat")
class Stat(PipelineNode):
    """
    Print schema, row count, and top rows.
    """

    def __init__(
        self,
        runtime,
        child_configs: list = None,
    ):
        super().__init__(runtime, child_configs)

    def run(self):
        ds = union_children(self.children, by_name=False)

        print("\n--- Dataset Statistics ---")
        print("\n--- 1. Dataset Schema ---")
        print(ds.schema())

        print("\n--- 2. Dataset Row Count ---")
        count = ds.count()
        print(f"Total Rows: {count}")

        print("\n--- 3. Dataset Head (Top 20) ---")
        show_dataset(ds, n=20, vertical=False)

        print("----------------------------\n")
        return ds
