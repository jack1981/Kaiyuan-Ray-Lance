"""Dataset preview node for lightweight inspection in pipeline graphs.

This node prints a sample of rows and passes data through unchanged.
"""

from datafiner.base import PipelineNode
from datafiner.dataset_utils import show_dataset, union_children
from datafiner.register import register


@register("Visualizer")
class Visualizer(PipelineNode):
    """
    A pipeline node to Visualize a specified number of rows from a DataFrame.
    """

    def __init__(
        self,
        runtime,
        child_configs: list = None,
        num_rows: int = 20,
        vertical: bool = False,
    ):
        """Configure preview row count and layout.

        Args:
            runtime: Shared runtime config.
            child_configs: Upstream node configs.
            num_rows: Number of rows to display.
            vertical: Whether to print row values vertically.

        Returns:
            None.

        Side effects:
            None during initialization.

        Assumptions:
            This operator is for diagnostics and does not mutate data.
        """
        super().__init__(runtime, child_configs)
        self.num_rows = num_rows
        self.vertical = vertical

    def run(self):
        """Print preview rows and return dataset unchanged.

        Inputs/outputs:
            Reads child dataset(s), prints sample, and returns original dataset.

        Side effects:
            Executes row sampling (`take`) and writes preview text to stdout.

        Assumptions:
            Sampling should not be used as a correctness gate.
        """
        ds = union_children(self.children, by_name=False)

        print(f"Visualizing the first {self.num_rows} rows:")
        show_dataset(ds, n=self.num_rows, vertical=self.vertical)

        return ds
