"""Global shuffle node for Ray datasets.

This node exists as a dedicated stage so configs can force full-row randomization
between deterministic transforms.
"""

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
        """Configure shuffle seed and child dependency.

        Args:
            runtime: Shared runtime config.
            seed: Optional seed for reproducible shuffle order.
            child_configs: Upstream node configs.

        Returns:
            None.

        Side effects:
            None during initialization.

        Assumptions:
            Global shuffle is acceptable from cost and memory perspective.
        """
        super().__init__(runtime, child_configs)
        self.seed = seed

    def run(self):
        """Shuffle all rows from child dataset output.

        Inputs/outputs:
            Reads child dataset(s) and returns shuffled dataset.

        Side effects:
            Triggers Ray shuffle operation across dataset blocks.

        Assumptions:
            Children are union-compatible by position.
        """
        ds = union_children(self.children, by_name=False)
        return ds.random_shuffle(seed=self.seed)
