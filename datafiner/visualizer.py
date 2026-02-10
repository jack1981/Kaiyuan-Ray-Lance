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
        """
        Initializes the Visualize node.

        Args:
            runtime: Ray runtime configuration.
            child_configs (list, optional): List of child node configurations. Defaults to None.
            num_rows (int, optional): The number of rows to visualize. Defaults to 20.
            vertical (bool, optional): Whether to display the output vertically. Defaults to False.
        """
        super().__init__(runtime, child_configs)
        self.num_rows = num_rows
        self.vertical = vertical

    def run(self):
        """
        Executes the visualization process.

        Returns:
            Ray dataset: The input dataset, passed through without modification.
        """
        ds = union_children(self.children, by_name=False)

        print(f"Visualizing the first {self.num_rows} rows:")
        show_dataset(ds, n=self.num_rows, vertical=self.vertical)

        return ds
