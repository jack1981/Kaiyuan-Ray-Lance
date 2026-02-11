"""Common schema/inspection utility nodes for pipeline composition.

These nodes provide lightweight column projection, renaming, and dataset
inspection helpers used across many example pipelines.
They are mostly plumbing operators that support reproducible YAML pipelines
without changing core quality/curriculum semantics.
See also `datafiner/dataset_utils.py` for shared batch transforms.
"""

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
        """Configure selected output columns.

        Args:
            runtime: Shared runtime config.
            select_cols: Non-empty ordered list of columns to keep.
            child_configs: Upstream node configs.

        Returns:
            None.

        Side effects:
            None.

        Assumptions:
            Requested columns exist in child dataset schemas.
        """
        super().__init__(runtime, child_configs)
        if not isinstance(select_cols, list) or not select_cols:
            raise ValueError("'select_cols' must be a non-empty list.")
        self.select_cols = select_cols

    def run(self):
        """Project child dataset to configured columns.

        Inputs/outputs:
            Reads child dataset(s) and returns projected dataset.

        Side effects:
            Prints selection info and executes batch projection transform.

        Assumptions:
            Children are union-compatible by position.
        """
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
        """Configure columns to remove.

        Args:
            runtime: Shared runtime config.
            drop_cols: Non-empty list of columns to drop.
            child_configs: Upstream node configs.

        Returns:
            None.

        Side effects:
            None.

        Assumptions:
            Missing columns are ignored during drop operation.
        """
        super().__init__(runtime, child_configs)
        if not isinstance(drop_cols, list) or not drop_cols:
            raise ValueError("'drop_cols' must be a non-empty list.")
        self.drop_cols = drop_cols

    def run(self):
        """Drop configured columns from child dataset output.

        Inputs/outputs:
            Reads child dataset(s) and returns dataset with columns removed.

        Side effects:
            Prints drop info and runs batch drop transform.

        Assumptions:
            Children are union-compatible by position.
        """
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
        """Configure a single-column rename operation.

        Args:
            runtime: Shared runtime config.
            input_col: Existing column name.
            output_col: Replacement column name.
            child_configs: Upstream node configs.

        Returns:
            None.

        Side effects:
            None.

        Assumptions:
            Rename should be no-op when input column is missing.
        """
        super().__init__(runtime, child_configs)
        self.input_col = input_col
        self.output_col = output_col

    def run(self):
        """Rename one column in each pandas batch.

        Inputs/outputs:
            Reads child dataset(s) and returns renamed dataset.

        Side effects:
            Executes a `map_batches` transform.

        Assumptions:
            Batch column order is preserved except renamed field.
        """
        ds = union_children(self.children, by_name=False)

        def alias_batch(batch):
            """Rename `input_col` to `output_col` within one batch.

            Args:
                batch: Source pandas batch.

            Returns:
                Batch with renamed column when present.

            Side effects:
                None.

            Assumptions:
                Missing `input_col` should not raise.
            """
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
        """Initialize schema inspection node.

        No node-specific init behavior is added beyond `PipelineNode`; this is
        intentionally concise to avoid duplicating shared init docs.
        """
        super().__init__(runtime, child_configs)

    def run(self):
        """Print schema and pass dataset through unchanged.

        Inputs/outputs:
            Reads child dataset(s) and returns same dataset object.

        Side effects:
            Calls `ds.schema()` and prints to stdout.

        Assumptions:
            Schema evaluation is acceptable for diagnostics in this node.
        """
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
        """Initialize row-count inspection node.

        Like `Schema`, this only wires base node configuration.
        """
        super().__init__(runtime, child_configs)

    def run(self):
        """Print total row count and pass dataset through unchanged.

        Inputs/outputs:
            Reads child dataset(s), counts rows, and returns original dataset.

        Side effects:
            Executes `ds.count()` and prints results.

        Assumptions:
            Full-count materialization cost is acceptable for this inspection
            operator.
        """
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
        """Initialize comprehensive stats node.

        No custom initialization beyond the shared `PipelineNode` setup.
        """
        super().__init__(runtime, child_configs)

    def run(self):
        """Print schema/count/head summary and return original dataset.

        Inputs/outputs:
            Reads child dataset(s), prints diagnostics, returns same dataset.

        Side effects:
            Executes schema inspection, row count, and sample display.

        Assumptions:
            Diagnostic calls may trigger expensive evaluation by design.
        """
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
