"""Union nodes that combine multiple child datasets by position or name.

These nodes are thin wrappers over shared union helpers in
`datafiner/dataset_utils.py`.
They are commonly used to compose multi-source dataset mixtures prior to
quality filtering, repetition, or curriculum ordering stages.
"""

from datafiner.base import PipelineNode
from datafiner.dataset_utils import union_children
from datafiner.register import register


@register("UnionByPosition")
class UnionByPosition(PipelineNode):
    """
    Unions all child DataFrames based on their column order.

    This node takes multiple child nodes and performs a positional union,
    which requires all inputs to have the same number of columns in the same order.
    """

    def __init__(
        self,
        runtime,
        child_configs: list = None,
    ):
        """Validate positional-union node configuration.

        Args:
            runtime: Shared runtime config.
            child_configs: Upstream node configs.

        Returns:
            None.

        Side effects:
            Instantiates child nodes through base class initialization.

        Assumptions:
            At least one child exists and all child outputs have compatible
            positional schemas.
        """
        super().__init__(runtime, child_configs)
        if not self.children or len(self.children) == 0:
            raise ValueError("UnionByPosition must have at least one child node.")

    def run(self):
        """Union all child datasets by positional schema.

        Inputs/outputs:
            Reads all child datasets and returns a single unioned dataset.

        Side effects:
            Executes child nodes and Ray union planning.

        Assumptions:
            Positional column counts match across children.
        """
        return union_children(self.children, by_name=False)


@register("UnionByName")
class UnionByName(PipelineNode):
    """
    Unions all child DataFrames based on their column names.

    This node unions child datasets by column names. Columns that do not
    exist in one dataset but exist in another are filled with null when enabled.
    """

    def __init__(
        self,
        runtime,
        allow_missing_columns: bool = False,
        child_configs: list = None,
    ):
        """Validate union-by-name configuration.

        Args:
            runtime: Shared runtime config.
            allow_missing_columns: Whether to null-fill absent columns across
                inputs.
            child_configs: Upstream node configs.

        Returns:
            None.

        Side effects:
            Instantiates child nodes through base class initialization.

        Assumptions:
            At least one child exists; schema differences are allowed only when
            `allow_missing_columns=True`.
        """
        super().__init__(runtime, child_configs)
        if not self.children or len(self.children) == 0:
            raise ValueError("UnionByName must have at least one child node.")
        self.allow_missing_columns = allow_missing_columns

    def run(self):
        """Union all child datasets by column names.

        Inputs/outputs:
            Reads all child datasets and returns one unioned dataset.

        Side effects:
            Executes child nodes and possible schema-alignment transforms.

        Assumptions:
            Name-based union semantics follow `allow_missing_columns` flag.
        """
        return union_children(
            self.children,
            by_name=True,
            allow_missing_columns=self.allow_missing_columns,
        )
