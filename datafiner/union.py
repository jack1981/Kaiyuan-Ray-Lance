from pyspark.sql import SparkSession, DataFrame
from datafiner.base import PipelineNode
from datafiner.register import register


@register("UnionByPosition")
class UnionByPosition(PipelineNode):
    """
    Unions all child DataFrames based on their column order.

    This node takes multiple child nodes and performs a standard
    Spark .union() operation, which requires that all DataFrames
    have the same number of columns in the same order.
    """

    def __init__(
        self,
        spark: SparkSession,
        child_configs: list = None,
    ):
        super().__init__(spark, child_configs)
        if not self.children or len(self.children) == 0:
            raise ValueError("UnionByPosition must have at least one child node.")

    def run(self) -> DataFrame:
        # Start with the first child's DataFrame
        df_union = self.children[0].run()

        # Union all other children
        if len(self.children) > 1:
            for child in self.children[1:]:
                df = child.run()

                # Optional: Add a check for schema mismatch
                if len(df.columns) != len(df_union.columns):
                    print(
                        f"Warning: UnionByPosition may fail. "
                        f"Schema 1 has {len(df_union.columns)} columns "
                        f"while Schema 2 has {len(df.columns)} columns."
                    )

                df_union = df_union.union(df)

        return df_union


@register("UnionByName")
class UnionByName(PipelineNode):
    """
    Unions all child DataFrames based on their column names.

    This node takes multiple child nodes and performs a .unionByName()
    operation. Columns that do not exist in one DataFrame but
    exist in another will be filled with null (if enabled).
    """

    def __init__(
        self,
        spark: SparkSession,
        allow_missing_columns: bool = False,
        child_configs: list = None,
    ):
        """
        :param allow_missing_columns: If True, allows unions between
                                      DataFrames with different sets of
                                      columns (missing cols will be null).
        """
        super().__init__(spark, child_configs)
        if not self.children or len(self.children) == 0:
            raise ValueError("UnionByName must have at least one child node.")
        self.allow_missing_columns = allow_missing_columns

    def run(self) -> DataFrame:
        # Start with the first child's DataFrame
        df_union = self.children[0].run()

        # Union all other children by name
        if len(self.children) > 1:
            for child in self.children[1:]:
                df = child.run()
                df_union = df_union.unionByName(
                    df, allowMissingColumns=self.allow_missing_columns
                )

        return df_union
