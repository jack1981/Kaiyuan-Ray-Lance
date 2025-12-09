from scipy.integrate import quad as integrate
import numpy as np
from typing import List, Text, Tuple
from itertools import tee
import re
import hashlib
import struct
import jieba

from pyspark.sql import SparkSession, DataFrame
from pyspark.sql import functions as F
from pyspark.sql.types import StringType

from datafiner.base import PipelineNode
from datafiner.deduplication.text_normalization import text_normalization
from datafiner.deduplication.wcc import weakly_connected_component_spark
from datafiner.register import register

MERSENNE_PRIME = np.uint64((1 << 61) - 1)
NON_ALPHA = re.compile("\W", re.UNICODE)


def ngrams(sequence: List[Text], n: int, min_length: int = 5):
    if len(sequence) < min_length:
        return []
    if len(sequence) < n:
        return [tuple(sequence)]
    iterables = tee(iter(sequence), n)
    for i, sub_iterable in enumerate(iterables):
        for _ in range(i):
            next(sub_iterable, None)
    return zip(*iterables)


def sha1_hash64(data: str):
    """A 64-bit hash function based on SHA1.

    Args:
        data (bytes): the data to generate 64-bit integer hash from.

    Returns:
        int: an integer hash value that can be encoded using 64 bits.
    """
    return struct.unpack("<Q", hashlib.sha1(data).digest()[:8])[0]


def generate_hash_values(
    uid: int,
    content: str,
    ngram_size: int,
    min_length: int,
    perm_a: np.ndarray,
    perm_b: np.ndarray,
    num_bands: int,
    num_rows_per_band: int,
    language: str = "en",
):
    if language == "en":
        tokens = {
            " ".join(t)
            for t in ngrams(NON_ALPHA.split(content), ngram_size, min_length)
        }
    else:
        words_no_space = [w for w in jieba.lcut(content) if w.strip() != ""]
        tokens = {" ".join(t) for t in ngrams(words_no_space, ngram_size, min_length)}
    if len(tokens) == 0:
        return []
    hv = np.fromiter(
        (sha1_hash64(token.encode("utf-8")) for token in tokens),
        dtype=np.uint64,
        count=len(tokens),
    ).reshape(-1, 1)
    phv = (hv * perm_a + perm_b) % MERSENNE_PRIME
    phv = np.min(phv, axis=0).astype(np.uint64)
    Hs = [
        bytes(phv[i * num_rows_per_band : (i + 1) * num_rows_per_band].byteswap().data)
        for i in range(num_bands)
    ]

    return [(i, Hs[i], uid) for i in range(num_bands)]


def generate_edges(nodes: List[int]) -> List[Tuple[int, int]]:
    nodes = list(nodes)
    if len(nodes) <= 1:
        return []

    min_node = min(nodes)
    return [(n, min_node) for n in nodes if n != min_node]


@register("MinHash")
class MinHash(PipelineNode):
    """
    MinHash is a family of algorithms for finding similar items in a dataset.
    It is based on the idea that two sets are similar if they have many elements in common.
    After MinHash, duplicated items are removed and each item is assigned with the number of copies.
    The number of copies (duplicate_count) is the number of times the item appears in the dataset.
    """

    def __init__(
        self,
        spark: SparkSession,
        num_permutations: int,
        threshold: float,
        num_parallel: int,
        language: str = "en",
        child_configs: list = None,
        input_col_name: str = "text",
        seed: int = 1234,
        ngram_size: int = 5,
        min_length: int = 5,
    ):
        super().__init__(spark, child_configs)
        self.num_permutations = num_permutations
        self.threshold = threshold
        self.num_parallel = num_parallel
        self.ngram_size = ngram_size
        self.min_length = min_length
        self.input_col_name = input_col_name

        assert language in ["en", "zh"], "language must be either 'en' or 'zh'"
        self.language = language

        self.optimal_bands_and_rows()

        gen = np.random.RandomState(seed)
        self.perm_a = gen.randint(
            1, MERSENNE_PRIME, size=(1, self.num_permutations), dtype=np.uint64
        )
        self.perm_b = gen.randint(
            1, MERSENNE_PRIME, size=(1, self.num_permutations), dtype=np.uint64
        )

    def optimal_bands_and_rows(
        self, false_positive_weight: float = 0.5, false_negative_weight: float = 0.5
    ):
        def false_positive_area(threshold: float, b: int, r: int):
            """Source: `datasketch.lsh`"""

            def area(s):
                return 1 - (1 - s ** float(r)) ** float(b)

            a, _ = integrate(area, 0.0, threshold)
            return a

        def false_negative_area(threshold: float, b: int, r: int):
            """Source: `datasketch.lsh`"""

            def area(s):
                return 1 - (1 - (1 - s ** float(r)) ** float(b))

            a, _ = integrate(area, threshold, 1.0)
            return a

        min_error = float("inf")
        opt = (0, 0)
        for b in range(1, self.num_permutations + 1):
            max_r = int(self.num_permutations / b)
            for r in range(1, max_r + 1):
                fp = false_positive_area(self.threshold, b, r)
                fn = false_negative_area(self.threshold, b, r)
                error = fp * false_positive_weight + fn * false_negative_weight
                if error < min_error:
                    min_error = error
                    opt = (b, r)
        self.num_bands = opt[0]
        self.num_rows_per_band = opt[1]
        self.num_permutations = self.num_bands * self.num_rows_per_band

    def sample_similar_content(self, edges: DataFrame, records: DataFrame) -> DataFrame:
        sample_edges = edges.sample(withReplacement=False, fraction=0.1)
        edges_with_src = sample_edges.join(
            records.withColumnRenamed("uid", "src").withColumnRenamed(
                "content", "content_src"
            ),
            on="src",
            how="left",
        )
        edges_with_both = edges_with_src.join(
            records.withColumnRenamed("uid", "dst").withColumnRenamed(
                "content", "content_dst"
            ),
            on="dst",
            how="left",
        )
        result = edges_with_both.select("content_src", "content_dst")
        result.show(truncate=50)

    def run(self):
        df = self.children[0].run()
        if len(self.children) > 1:
            for child in self.children[1:]:
                df = df.union(child.run())
        return self.minhash(df)

    def minhash(
        self,
        df: DataFrame,
    ) -> DataFrame:
        # create uid
        df = df.filter(
            (F.col(self.input_col_name).isNotNull())
            & (F.col(self.input_col_name) != "")
        )
        uid_df = df.withColumn("uid", F.monotonically_increasing_id())
        num_doc = uid_df.count()

        # Check if duplicate_count column already exists, if not create it with value 1
        if "duplicate_count" not in uid_df.columns:
            uid_df = uid_df.withColumn("duplicate_count", F.lit(1.0))

        # normalize the text
        normal_str = F.udf(text_normalization, StringType())
        normal_df = uid_df.withColumn("content", normal_str(F.col(self.input_col_name)))
        records = normal_df.select("uid", "content")
        records.show()

        # generate hash values
        perm_a = self.perm_a
        perm_b = self.perm_b
        ngram_size = self.ngram_size
        min_length = self.min_length
        num_bands = self.num_bands
        num_rows_per_band = self.num_rows_per_band
        language = self.language
        # generate hash values
        hash_rdd = records.rdd.flatMap(
            lambda x: generate_hash_values(
                uid=x["uid"],
                content=x["content"],
                ngram_size=ngram_size,
                min_length=min_length,
                perm_a=perm_a,
                perm_b=perm_b,
                num_bands=num_bands,
                num_rows_per_band=num_rows_per_band,
                language=language,
            )
        )

        # generate edges
        same_samples = hash_rdd.groupBy(lambda x: (x[0], x[1]))
        edges = same_samples.flatMap(
            lambda x: generate_edges(i[2] for i in x[1])
        ).distinct()
        components = self.spark.createDataFrame(edges, ["src", "dst"])

        # wcc
        wcc = weakly_connected_component_spark(components, self.num_parallel)
        wcc = wcc.withColumn(
            "component",
            F.when(F.col("component") > F.col("vid"), F.col("vid")).otherwise(
                F.col("component")
            ),
        )
        wcc_filter = wcc.filter(F.col("vid") > F.col("component"))
        records_filtered = uid_df.join(
            wcc_filter, uid_df.uid == wcc_filter.vid, "left_anti"
        )
        filtered_count = records_filtered.count()
        print(
            f"whole doc: {num_doc}, dup doc: {num_doc - filtered_count}, "
            f"duplicate rate: {(num_doc - filtered_count) * 100 / num_doc}%"
        )

        # Sum up duplicate_count for each component instead of just counting rows
        # First, join the WCC results with original data to get duplicate_count values
        wcc_with_counts = wcc.join(
            uid_df.select("uid", "duplicate_count"), wcc.vid == uid_df.uid, "inner"
        )

        # Group by component and sum the duplicate_count values
        wcc_group = wcc_with_counts.groupBy("component").agg(
            F.sum("duplicate_count").alias("accumulated_duplicate_count")
        )

        # Join with records_filtered and update duplicate_count
        records_filtered = records_filtered.join(
            wcc_group, records_filtered.uid == wcc_group.component, "left"
        )

        # Use accumulated count if available, otherwise keep original duplicate_count
        records_filtered = records_filtered.withColumn(
            "duplicate_count",
            F.coalesce(F.col("accumulated_duplicate_count"), F.col("duplicate_count")),
        ).drop("component", "accumulated_duplicate_count")

        records_filtered = records_filtered.drop("uid")

        return records_filtered
