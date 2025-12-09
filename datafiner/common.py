from pyspark.sql import SparkSession, DataFrame
from datafiner.base import PipelineNode
from datafiner.register import register
import pyspark.sql.functions as F


@register("ColumnSelect")
class ColumnSelect(PipelineNode):
    """
    A pipeline node to select a specific set of columns,
    dropping all others.
    """

    def __init__(
        self,
        spark: SparkSession,
        select_cols: list,
        child_configs: list = None,
    ):
        """
        Initializes the ColumnSelect node.

        Args:
            spark (SparkSession): The Spark session object.
            select_cols (list): A list of column names to keep.
            child_configs (list, optional): List of child node configurations.
        """
        super().__init__(spark, child_configs)
        if not isinstance(select_cols, list) or not select_cols:
            raise ValueError("'select_cols' must be a non-empty list.")
        self.select_cols = select_cols

    def run(self):
        df = self.children[0].run()
        if len(self.children) > 1:
            for child in self.children[1:]:
                df = df.union(child.run())

        print(f"[ColumnSelect] Selecting columns: {self.select_cols}")
        return df.select(*self.select_cols)


@register("ColumnDrop")
class ColumnDrop(PipelineNode):
    """
    A pipeline node to drop a specific set of columns,
    keeping all others.
    """

    def __init__(
        self,
        spark: SparkSession,
        drop_cols: list,
        child_configs: list = None,
    ):
        """
        Initializes the ColumnDrop node.

        Args:
            spark (SparkSession): The Spark session object.
            drop_cols (list): A list of column names to drop.
            child_configs (list, optional): List of child node configurations.
        """
        super().__init__(spark, child_configs)
        if not isinstance(drop_cols, list) or not drop_cols:
            raise ValueError("'drop_cols' must be a non-empty list.")
        self.drop_cols = drop_cols

    def run(self):
        df = self.children[0].run()
        if len(self.children) > 1:
            for child in self.children[1:]:
                df = df.union(child.run())

        print(f"[ColumnDrop] Dropping columns: {self.drop_cols}")
        return df.drop(*self.drop_cols)


@register("ColumnAlias")
class ColumnAlias(PipelineNode):
    def __init__(
        self,
        spark: SparkSession,
        input_col: str,
        output_col: str,
        child_configs: list = None,
    ):
        super().__init__(spark, child_configs)
        self.input_col = input_col
        self.output_col = output_col

    def run(self):
        df = self.children[0].run()
        if len(self.children) > 1:
            for child in self.children[1:]:
                df = df.union(child.run())
        return df.withColumnRenamed(self.input_col, self.output_col)


@register("Schema")
class Schema(PipelineNode):
    """
    print schema
    """

    def __init__(
        self,
        spark: SparkSession,
        child_configs: list = None,
    ):
        super().__init__(spark, child_configs)

    def run(self) -> DataFrame:
        # Get DataFrame from child node(s)
        df = self.children[0].run()
        if len(self.children) > 1:
            for child in self.children[1:]:
                df = df.union(child.run())

        print("\n--- DataFrame Schema ---")
        df.printSchema()
        print("------------------------------\n")

        # Return the DataFrame unchanged
        return df


@register("Row Number")
class RowNumber(PipelineNode):
    """
    print row number
    """

    def __init__(
        self,
        spark: SparkSession,
        child_configs: list = None,
    ):
        super().__init__(spark, child_configs)

    def run(self) -> DataFrame:
        # Get DataFrame from child node(s)
        df = self.children[0].run()
        if len(self.children) > 1:
            for child in self.children[1:]:
                df = df.union(child.run())

        # Perform the count action
        count = df.count()

        print(f"\n--- DataFrame Row Count ---")
        print(f"Total Rows: {count}")
        print("-------------------------------\n")

        # Return the DataFrame unchanged
        return df


@register("Stat")
class Stat(PipelineNode):
    """
    1. show schema
    2. print row number
    3. show head, top 20 row, should at least show column titles for each column.
    """

    def __init__(
        self,
        spark: SparkSession,
        child_configs: list = None,
    ):
        super().__init__(spark, child_configs)

    def run(self) -> DataFrame:
        # Get DataFrame from child node(s)
        df = self.children[0].run()
        if len(self.children) > 1:
            for child in self.children[1:]:
                df = df.union(child.run())

        # Cache the DataFrame since we are performing multiple actions
        df.cache()

        print("\n--- DataFrame Statistics ---")

        # 1. show schema
        print("\n--- 1. DataFrame Schema ---")
        df.printSchema()

        # 2. print row number
        print("\n--- 2. DataFrame Row Count ---")
        count = df.count()
        print(f"Total Rows: {count}")

        # 3. show head, top 20 row
        print("\n--- 3. DataFrame Head (Top 20) ---")
        # df.show() defaults to 20 rows and includes column titles
        df.show(20)

        print("----------------------------------\n")

        # Unpersist the DataFrame after use
        df.unpersist()

        # Return the original DataFrame for the next pipeline step
        return df
