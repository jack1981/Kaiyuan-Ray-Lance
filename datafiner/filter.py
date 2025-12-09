from datafiner.base import PipelineNode
from datafiner.register import register
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql import functions as F
from enum import Enum
import operator


class ComparisonOperator(Enum):
    """
    Enumeration for comparison operators to provide a robust way to handle comparisons.
    Each member has a symbolic representation and the corresponding function from the operator module.
    """

    LARGER = (">", operator.gt)
    SMALLER = ("<", operator.lt)
    LARGER_OR_EQUAL = (">=", operator.ge)
    SMALLER_OR_EQUAL = ("<=", operator.le)
    EQUAL = ("==", operator.eq)

    def __init__(self, symbol, func):
        self.symbol = symbol
        self.func = func

    @classmethod
    def from_str(cls, s: str):
        """
        Converts a user-friendly string to a ComparisonOperator enum member.
        This makes the input flexible while keeping the internal logic robust.

        Args:
            s (str): The string representation of the operator (e.g., 'larger', '>', '<=').

        Returns:
            ComparisonOperator: The corresponding enum member.
        """
        s_lower = s.lower().strip()
        if s_lower in ("larger", ">"):
            return cls.LARGER
        if s_lower in ("smaller", "<"):
            return cls.SMALLER
        if s_lower in ("larger_or_equal", ">="):
            return cls.LARGER_OR_EQUAL
        if s_lower in ("smaller_or_equal", "<="):
            return cls.SMALLER_OR_EQUAL
        if s_lower in ("equal", "=="):
            return cls.EQUAL
        raise ValueError(f"Unsupported comparison operator: {s}")


@register("Filter")
class Filter(PipelineNode):
    """
    A pipeline node to filter a DataFrame based on a column's value against a static threshold.
    """

    def __init__(
        self,
        spark: SparkSession,
        child_configs: list = None,
        column: str = None,
        comparison: str = "larger",
        threshold: float = 0.0,
    ):
        """
        Initializes the Filter node.

        Args:
            spark (SparkSession): The Spark session object.
            child_configs (list, optional): List of child node configurations. Defaults to None.
            column (str): The name of the column to filter on.
            comparison (str, optional): The comparison operator (e.g., 'larger', '<', 'smaller_or_equal').
                                      Defaults to 'larger'.
            threshold (float, optional): The value to compare against. Defaults to 0.0.
        """
        super().__init__(spark, child_configs)
        if column is None:
            raise ValueError(
                "The 'column' argument must be specified for the Filter node."
            )
        self.column = column
        self.comp_op = ComparisonOperator.from_str(comparison)
        self.threshold = threshold

    def run(self) -> DataFrame:
        """
        Executes the filtering process.

        Returns:
            DataFrame: The filtered DataFrame.
        """
        df = self.children[0].run()
        if len(self.children) > 1:
            for child in self.children[1:]:
                df = df.union(child.run())

        # Use the function from the Enum to create the filter condition
        filter_condition = self.comp_op.func(F.col(self.column), self.threshold)
        return df.filter(filter_condition)


@register("FilterByRatio")
class FilterByRatio(PipelineNode):
    """
    A pipeline node to filter a DataFrame to keep a certain ratio of rows
    based on the values in a specified column. For example, keeping the top 10% of rows
    based on a 'score' column.
    """

    def __init__(
        self,
        spark: SparkSession,
        child_configs: list = None,
        column: str = None,
        comparison: str = "larger",
        keep_ratio: float = 0.1,
        quantile_error: float = 1e-4,
    ):
        """
        Initializes the FilterByRatio node.

        Args:
            spark (SparkSession): The Spark session object.
            child_configs (list, optional): List of child node configurations. Defaults to None.
            column (str): The name of the column to use for determining the filtering threshold.
            comparison (str, optional): 'larger' to keep the top ratio of values,
                                        'smaller' to keep the bottom ratio. Defaults to 'larger'.
            keep_ratio (float, optional): The ratio of rows to keep (between 0.0 and 1.0). Defaults to 0.1.
        """
        super().__init__(spark, child_configs)
        if not (0.0 <= keep_ratio <= 1.0):
            raise ValueError("keep_ratio must be between 0.0 and 1.0")
        if column is None:
            raise ValueError(
                "The 'column' argument must be specified for the Filter node."
            )
        self.column = column
        self.comp_op = ComparisonOperator.from_str(comparison)
        self.keep_ratio = keep_ratio
        self.quantile_error = quantile_error

    def run(self) -> DataFrame:
        """
        Calculates a dynamic threshold based on the keep_ratio and filters the DataFrame.

        Returns:
            DataFrame: The filtered DataFrame containing the desired ratio of rows.
        """
        df = self.children[0].run()
        if len(self.children) > 1:
            for child in self.children[1:]:
                df = df.union(child.run())

        # Determine the quantile needed to find the correct threshold value.
        if self.comp_op == ComparisonOperator.LARGER:
            # To keep the largest 10% (keep_ratio=0.1) of rows, we need to find the value
            # at the 90th percentile (quantile=0.9) and keep all rows with a value larger than that.
            quantile = 1.0 - self.keep_ratio
        elif self.comp_op == ComparisonOperator.SMALLER:
            # To keep the smallest 10% (keep_ratio=0.1), we need the value at the
            # 10th percentile (quantile=0.1) and keep all rows with a value smaller than that.
            quantile = self.keep_ratio
        else:
            raise ValueError(
                "FilterByRatio only supports 'larger' or 'smaller' comparisons."
            )

        temp_col_name = f"__temp_quantile_col_{self.column.replace('.', '_')}"
        df_with_temp_col = df.withColumn(temp_col_name, F.col(self.column))

        # approxQuantile computes the approximate quantiles of a numerical column.
        # It returns a list of values for the given quantiles.
        # The third parameter is the relative error tolerance (0.0 for exact results).
        # thresholds = df.approxQuantile(self.column, [quantile], 1e-8)
        # for speed, set error to 1e-4
        thresholds = df_with_temp_col.approxQuantile(
            temp_col_name, [quantile], self.quantile_error
        )
        self.threshold = thresholds[0]

        # With the dynamically calculated threshold, we can now perform the filter.
        filter_condition = self.comp_op.func(F.col(self.column), self.threshold)
        return df.filter(filter_condition)
