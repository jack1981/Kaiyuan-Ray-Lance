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
        super().__init__(runtime, child_configs)
        if not self.children or len(self.children) == 0:
            raise ValueError("UnionByPosition must have at least one child node.")

    def run(self):
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
        """
        :param allow_missing_columns: If True, allows unions between
                                      DataFrames with different sets of
                                      columns (missing cols will be null).
        """
        super().__init__(runtime, child_configs)
        if not self.children or len(self.children) == 0:
            raise ValueError("UnionByName must have at least one child node.")
        self.allow_missing_columns = allow_missing_columns

    def run(self):
        return union_children(
            self.children,
            by_name=True,
            allow_missing_columns=self.allow_missing_columns,
        )
