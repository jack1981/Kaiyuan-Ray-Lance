from pyspark.sql import SparkSession, DataFrame, Row
from pyspark.sql.types import StructType, StructField, StringType, ArrayType, DoubleType
from abc import ABC, abstractmethod
from typing import Union

import io
import numpy as np

from datafiner.base import PipelineNode
from datafiner.register import register
from pyspark.sql.types import *


def parse_spark_type(field):
    """Parse both primitive and complex (array, struct) types from YAML."""

    primitive_types = {
        "string": StringType(),
        "long": LongType(),
        "int": IntegerType(),
        "boolean": BooleanType(),
        "double": DoubleType(),
        "float": FloatType(),
        "binary": BinaryType(),
    }
    print(field)
    print(type(field))
    print(isinstance(field, str))
    print(field["type"] not in primitive_types)
    print("string" not in primitive_types)

    # ---- Case 1: primitive type as string ----
    if field.get("type", None) in primitive_types:
        return primitive_types[field["type"]]

    # ---- Case 2: array type ----
    if field["type"] == "array":
        if "elementType" not in field:
            raise ValueError(f"Array type must include elementType: {field}")
        element_type = parse_spark_type(field["elementType"])
        return ArrayType(element_type)

    # ---- Case 3: struct type ----
    if field["type"] == "struct":
        return StructType(
            [
                StructField(
                    f["name"], parse_spark_type(f), bool(f.get("nullable", True))
                )
                for f in field["fields"]
            ]
        )

    raise ValueError(f"Unsupported type definition: {field}")


def build_schema(fields_yaml):
    struct_fields = []
    for f in fields_yaml:
        print(f)
        struct_fields.append(
            StructField(f["name"], parse_spark_type(f), bool(f.get("nullable", True)))
        )
    return StructType(struct_fields)


class DataReader(PipelineNode, ABC):
    def __init__(
        self,
        spark: SparkSession,
        input_path: Union[str, list],
        num_parallel: int = None,
        child_configs: list = None,
        select_cols: list = None,
    ):
        super().__init__(spark, child_configs)
        self.input_path = input_path
        self.num_parallel = num_parallel
        self.select_cols = select_cols

    @abstractmethod
    def read(self) -> DataFrame:
        pass

    def run(self):
        assert len(self.children) == 0, "DataReader does not support child configs"
        return self.read()


@register("ParquetReader")
class ParquetReader(DataReader):
    def __init__(
        self,
        spark: SparkSession,
        input_path: Union[str, list],
        num_parallel: int = None,
        child_configs: list = None,
        select_cols: list = None,
        mergeSchema: str = "true",
        schema: Union[StructType, list, None] = None,  # Now supports YAML list
        datetimeRebaseModeInRead: str = "CORRECTED",
    ):
        super().__init__(spark, input_path, num_parallel, child_configs, select_cols)
        self.mergeSchema = mergeSchema
        self.datetimeRebaseModeInRead = datetimeRebaseModeInRead

        # If schema is provided as YAML list → convert to StructType
        if isinstance(schema, list):
            self.schema = build_schema(schema)
        else:
            self.schema = schema  # StructType or None

    def read(self) -> DataFrame:
        reader = self.spark.read.option("mergeSchema", self.mergeSchema)

        if self.schema is not None:
            reader = reader.schema(self.schema)

        if isinstance(self.input_path, list):
            df = reader.parquet(*self.input_path)
        else:
            df = reader.parquet(self.input_path)

        if self.select_cols is not None:
            df = df.select(self.select_cols)

        print(f"Number of rows: {df.count()}")

        if self.num_parallel is not None:
            df = df.repartition(self.num_parallel)

        return df


@register("ParquetReaderZstd")
class ParquetReaderZstd(DataReader):
    """
    Parquet reader optimized for reading Zstd-compressed files.
    Uses PyArrow for efficient decompression.
    """

    def __init__(
        self,
        spark: SparkSession,
        input_path: Union[str, list],
        num_parallel: int = None,
        child_configs: list = None,
        select_cols: list = None,
        mergeSchema: str = "true",
        schema: Union[StructType, list, None] = None,
        use_pyarrow: bool = True,
    ):
        super().__init__(spark, input_path, num_parallel, child_configs, select_cols)
        self.mergeSchema = mergeSchema
        self.use_pyarrow = use_pyarrow

        if isinstance(schema, list):
            self.schema = build_schema(schema)
        else:
            self.schema = schema

    def read(self) -> DataFrame:
        reader = self.spark.read.option("mergeSchema", self.mergeSchema)

        # Enable PyArrow for better Zstd decompression performance
        if self.use_pyarrow:
            reader = reader.option("spark.sql.execution.arrow.pyspark.enabled", "true")

        if self.schema is not None:
            reader = reader.schema(self.schema)

        if isinstance(self.input_path, list):
            df = reader.parquet(*self.input_path)
        else:
            df = reader.parquet(self.input_path)

        if self.select_cols is not None:
            df = df.select(self.select_cols)

        print(f"Number of rows: {df.count()}")

        if self.num_parallel is not None:
            df = df.repartition(self.num_parallel)

        return df


@register("JsonlZstReader")
class JsonlZstReader(DataReader):
    def __init__(
        self,
        spark: SparkSession,
        input_path: Union[str, list],
        num_parallel: int = None,
        child_configs: list = None,
        select_cols: list = None,
    ):
        super().__init__(spark, input_path, num_parallel, child_configs, select_cols)

    def read(self) -> DataFrame:
        if isinstance(self.input_path, list):
            df = self.spark.read.json(*self.input_path)
        else:
            df = self.spark.read.json(self.input_path)
        if self.select_cols is not None:
            df = df.select(self.select_cols)
        if self.num_parallel is not None:
            df = df.repartition(self.num_parallel)
        return df


@register("JsonReader")
class JsonReader(DataReader):
    """
    Reads JSON or JSON Lines files from HDFS or other file systems into a Spark DataFrame.
    """

    def __init__(
        self,
        spark: SparkSession,
        input_path: Union[str, list],
        multiLine: bool = False,  # Option to handle multi-line JSON objects
        num_parallel: int = None,
        child_configs: list = None,
        select_cols: list = None,
    ):
        super().__init__(spark, input_path, num_parallel, child_configs, select_cols)
        self.multiLine = multiLine

    def read(self) -> DataFrame:
        print(
            f"INFO: Reading JSON from path: {self.input_path} with multiLine={self.multiLine}"
        )

        # Use Spark's native JSON reader, which is HDFS-compatible
        reader = self.spark.read.option("multiLine", self.multiLine)

        if isinstance(self.input_path, list):
            df = reader.json(*self.input_path)
        else:
            df = reader.json(self.input_path)

        if self.select_cols is not None:
            df = df.select(self.select_cols)

        if self.num_parallel is not None:
            df = df.repartition(self.num_parallel)

        return df


@register("FormatReader")
class FormatReader(DataReader):
    """
    A generic reader that can read from parquet, JSON, or other formats based on the provided type.
    """

    def __init__(
        self,
        spark: SparkSession,
        input_path: Union[str, list],
        data_format: str = "parquet",  # "json"
        num_parallel: int = None,
        child_configs: list = None,
        select_cols: list = None,
    ):
        super().__init__(spark, input_path, num_parallel, child_configs, select_cols)
        self.data_format = data_format.lower()

    def read(self) -> DataFrame:
        if isinstance(self.input_path, list):
            df = self.spark.read.format(self.data_format).load(*self.input_path)
        else:
            df = self.spark.read.format(self.data_format).load(self.input_path)
        if self.select_cols is not None:
            df = df.select(self.select_cols)
        if self.num_parallel is not None:
            df = df.repartition(self.num_parallel)
        return df


@register("NpyReader")
class NpyReader(DataReader):
    def __init__(
        self,
        spark: SparkSession,
        input_path: Union[str, list],
        column_name: str,
        path_column: str = None,
        num_parallel: int = None,
        child_configs: list = None,
        select_cols: list = None,
    ):
        super().__init__(spark, input_path, num_parallel, child_configs, select_cols)
        if not column_name:
            raise ValueError("'column_name' must be a non-empty string.")
        self.column_name = column_name
        self.path_column = path_column

    def read(self) -> DataFrame:
        def _parse_npy_content(file_tuple: tuple) -> Union[Row, None]:
            filepath, content_stream = file_tuple
            try:
                content_bytes = content_stream.read()
                with io.BytesIO(content_bytes) as bio:
                    numpy_array = np.load(bio)

                data_list = numpy_array.tolist()

                row_data = {self.column_name: data_list}
                if self.path_column:
                    row_data[self.path_column] = filepath
                return Row(**row_data)
            except Exception as e:
                print(f"Warning: Could not read or parse npy file '{filepath}': {e}")
                return None

        schema_fields = []
        if self.path_column:
            schema_fields.append(
                StructField(self.path_column, StringType(), nullable=False)
            )
        schema_fields.append(
            StructField(self.column_name, ArrayType(DoubleType()), nullable=True)
        )
        schema = StructType(schema_fields)

        binary_rdd = self.spark.sparkContext.binaryFiles(self.input_path)
        parsed_rdd = binary_rdd.map(_parse_npy_content).filter(lambda x: x is not None)

        if parsed_rdd.isEmpty():
            print(f"Warning: No valid npy files found in '{self.input_path}'")
            df = self.spark.createDataFrame(self.spark.sparkContext.emptyRDD(), schema)
        else:
            df = self.spark.createDataFrame(parsed_rdd, schema)

        if self.select_cols is not None:
            df = df.select(self.select_cols)

        if self.num_parallel is not None:
            df = df.repartition(self.num_parallel)

        return df
