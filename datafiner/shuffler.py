from __future__ import annotations

from datafiner.base import PipelineNode
from datafiner.dataset_utils import union_children
from datafiner.register import register


@register("Shuffler")
class Shuffler(PipelineNode):
    """
    A pipeline node to globally shuffle all rows of a DataFrame.
    """

    def __init__(
        self,
        runtime,
        seed: int | None = None,
        child_configs: list = None,
    ):
        """
        Initializes the Shuffle node.

        Args:
            runtime: Ray runtime configuration.
            child_configs (list, optional): List of child node configurations. Defaults to None.
        """
        super().__init__(runtime, child_configs)
        self.seed = seed

    def run(self):
        """
        Executes the global shuffle.

        Returns:
            Ray dataset: The shuffled dataset.
        """
        ds = union_children(self.children, by_name=False)
        return ds.random_shuffle(seed=self.seed)
