from abc import ABC, abstractmethod
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.functions import rand
from transformers import AutoTokenizer

from datafiner.base import PipelineNode
from datafiner.register import register


def _as_lance_identifier(path: str) -> str:
    if path.startswith("lance."):
        return path
    return f"lance.`{path.replace('`', '``')}`"


class DataWriter(PipelineNode, ABC):
    def __init__(
        self,
        spark: SparkSession,
        output_path: str,
        shuffle: bool = True,
        select_cols: list = None,
        child_configs: list = None,
    ):
        super().__init__(spark, child_configs)
        self.output_path = output_path
        self.shuffle = shuffle
        self.select_cols = select_cols

    @abstractmethod
    def write(self, df: DataFrame):
        pass

    def run(self):
        df_union = self.children[0].run()
        if len(self.children) > 1:
            for child in self.children[1:]:
                df = child.run()
                df_union = df_union.union(df)
        if self.select_cols:
            df_union = df_union.select(*self.select_cols)
        if self.shuffle:
            df_union = df_union.orderBy(rand())
        return self.write(df_union)


@register("ParquetWriter")
class ParquetWriter(DataWriter):
    def __init__(
        self,
        spark: SparkSession,
        output_path: str,
        shuffle: bool = False,
        num_output_files: int = None,
        num_read_partitions: int = None,
        mode: str = "overwrite",
        select_cols: list = None,
        child_configs: list = None,
    ):
        super().__init__(spark, output_path, shuffle, select_cols, child_configs)
        self.num_output_files = num_output_files
        self.num_read_partitions = num_read_partitions
        self.mode = mode

    def _path_exists_and_complete(self) -> bool:
        """
        Checks if the target output path exists and is non-empty.
        """
        fs = self.spark.sparkContext._jvm.org.apache.hadoop.fs.FileSystem.get(
            self.spark.sparkContext._jsc.hadoopConfiguration()
        )
        hadoop_output_path = self.spark.sparkContext._jvm.org.apache.hadoop.fs.Path(
            self.output_path
        )
        if not fs.exists(hadoop_output_path):
            return False
        return len(fs.listStatus(hadoop_output_path)) > 0

    def run(self) -> DataFrame:
        """
        Runs the writer. If mode is 'read_if_exists', checks for a cached
        version before executing the child pipeline.
        """
        # 1. Check cache *before* running any children
        if self.mode == "read_if_exists":
            print(
                f"[ParquetWriter] Mode 'read_if_exists' set. Checking cache: {self.output_path}"
            )

            # This check will now raise an error if FS fails
            if self._path_exists_and_complete():
                # 2. If HIT: Read from disk and return.
                print(f"[ParquetWriter] Cache hit. Reading from path.")

                # Read the DataFrame from the cached path
                df_read = self.spark.read.table(_as_lance_identifier(self.output_path))

                if self.num_read_partitions is not None:
                    print(
                        f"[ParquetWriter] Repartitioning read DataFrame to {self.num_read_partitions} partitions."
                    )
                    df_read = df_read.repartition(self.num_read_partitions)

                return df_read
            else:
                print(f"[ParquetWriter] Cache miss. Proceeding to compute and write.")

        print("[ParquetWriter] Computing DataFrame from children...")
        df_union = self.children[0].run()
        if len(self.children) > 1:
            for child in self.children[1:]:
                df = child.run()
                df_union = df_union.union(df)

        if self.select_cols:
            df_union = df_union.select(*self.select_cols)
        if self.shuffle:
            df_union = df_union.orderBy(rand())

        # 5. Pass the computed DataFrame to the write method to be saved.
        return self.write(df_union)

    def write(self, df: DataFrame):
        """
        Writes the given DataFrame to parquet.
        Assumes the decision to write (vs. read from cache)
        has already been made by the 'run()' method.
        """

        # Determine the final write mode.
        if self.mode == "read_if_exists":
            final_write_mode = "overwrite"
        else:
            final_write_mode = self.mode

        # 2. Repartition (This is for *writing*)
        if self.num_output_files is not None:
            print(
                f"[ParquetWriter] Repartitioning DataFrame to {self.num_output_files} partitions."
            )
            df = df.repartition(self.num_output_files)

        # 3. Perform the write
        print(
            f"[ParquetWriter] Writing data to {self.output_path} (Mode: '{final_write_mode}')"
        )
        lance_identifier = _as_lance_identifier(self.output_path)
        writer_v2 = df.writeTo(lance_identifier)
        if final_write_mode in ("overwrite", "read_if_exists"):
            writer_v2.using("lance").createOrReplace()
        elif final_write_mode == "append":
            writer_v2.append()
        elif final_write_mode == "ignore":
            try:
                writer_v2.using("lance").create()
            except Exception as exc:
                if "already exists" not in str(exc).lower():
                    raise
        else:
            writer_v2.using("lance").create()

        return df


@register("ParquetWriterZstd")
class ParquetWriterZstd(DataWriter):
    """
    Parquet writer with Zstd compression at level 9.
    """

    def __init__(
        self,
        spark: SparkSession,
        output_path: str,
        shuffle: bool = False,
        num_output_files: int = None,
        num_read_partitions: int = None,
        mode: str = "overwrite",
        select_cols: list = None,
        child_configs: list = None,
        compression_level: int = 9,
        use_coalesce: bool = False,
        merge_count: int = 128,
    ):
        super().__init__(spark, output_path, shuffle, select_cols, child_configs)
        self.num_output_files = num_output_files
        self.num_read_partitions = num_read_partitions
        self.mode = mode
        self.compression_level = compression_level
        self.use_coalesce = use_coalesce
        self.merge_count = merge_count

    def _path_exists_and_complete(self) -> bool:
        fs = self.spark.sparkContext._jvm.org.apache.hadoop.fs.FileSystem.get(
            self.spark.sparkContext._jsc.hadoopConfiguration()
        )
        hadoop_output_path = self.spark.sparkContext._jvm.org.apache.hadoop.fs.Path(
            self.output_path
        )
        if not fs.exists(hadoop_output_path):
            return False
        return len(fs.listStatus(hadoop_output_path)) > 0

    def run(self) -> DataFrame:
        if self.mode == "read_if_exists":
            print(
                f"[ParquetWriterZstd] Mode 'read_if_exists' set. Checking cache: {self.output_path}"
            )

            if self._path_exists_and_complete():
                print(f"[ParquetWriterZstd] Cache hit. Reading from path.")
                df_read = self.spark.read.table(_as_lance_identifier(self.output_path))

                if self.num_read_partitions is not None:
                    print(
                        f"[ParquetWriterZstd] Repartitioning read DataFrame to {self.num_read_partitions} partitions."
                    )
                    df_read = df_read.repartition(self.num_read_partitions)

                return df_read
            else:
                print(
                    f"[ParquetWriterZstd] Cache miss. Proceeding to compute and write."
                )

        print("[ParquetWriterZstd] Computing DataFrame from children...")
        df_union = self.children[0].run()
        if len(self.children) > 1:
            for child in self.children[1:]:
                df = child.run()
                df_union = df_union.union(df)

        if self.select_cols:
            df_union = df_union.select(*self.select_cols)
        if self.shuffle:
            df_union = df_union.orderBy(rand())

        return self.write(df_union)

    def write(self, df: DataFrame):
        if self.mode == "read_if_exists":
            final_write_mode = "overwrite"
        else:
            final_write_mode = self.mode

        if self.use_coalesce:
            if self.num_output_files is not None:
                df = df.coalesce(self.num_output_files)
            else:
                current_partitions = df.rdd.getNumPartitions()
                target_partitions = max(1, current_partitions // self.merge_count)
                df = df.coalesce(target_partitions)
        else:
            if self.num_output_files is not None:
                df = df.repartition(self.num_output_files)

        print(
            f"[ParquetWriterZstd] Writing data in Lance format to {self.output_path}"
        )

        lance_identifier = _as_lance_identifier(self.output_path)
        writer_v2 = df.writeTo(lance_identifier)
        if final_write_mode in ("overwrite", "read_if_exists"):
            writer_v2.using("lance").createOrReplace()
        elif final_write_mode == "append":
            writer_v2.append()
        elif final_write_mode == "ignore":
            try:
                writer_v2.using("lance").create()
            except Exception as exc:
                if "already exists" not in str(exc).lower():
                    raise
        else:
            writer_v2.using("lance").create()

        return df


@register("LanceWriter")
class LanceWriter(ParquetWriter):
    pass
