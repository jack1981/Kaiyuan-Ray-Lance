from __future__ import annotations

import hashlib
import re
import struct
from itertools import tee
from typing import List, Text, Tuple

import jieba
import numpy as np
from scipy.integrate import quad as integrate

from datafiner.base import PipelineNode
from datafiner.dataset_utils import dataset_from_pandas, union_children
from datafiner.deduplication.text_normalization import text_normalization
from datafiner.deduplication.wcc import weakly_connected_component
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


def sha1_hash64(data: bytes):
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
    hashes = [
        bytes(phv[i * num_rows_per_band : (i + 1) * num_rows_per_band].byteswap().data)
        for i in range(num_bands)
    ]

    return [(i, hashes[i], uid) for i in range(num_bands)]


def generate_edges(nodes: List[int]) -> List[Tuple[int, int]]:
    nodes = list(nodes)
    if len(nodes) <= 1:
        return []

    min_node = min(nodes)
    return [(n, min_node) for n in nodes if n != min_node]


@register("MinHash")
class MinHash(PipelineNode):
    """
    MinHash near-duplicate removal implemented on Ray datasets.
    """

    def __init__(
        self,
        runtime,
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
        super().__init__(runtime, child_configs)
        self.num_permutations = num_permutations
        self.threshold = threshold
        self.num_parallel = num_parallel
        self.ngram_size = ngram_size
        self.min_length = min_length
        self.input_col_name = input_col_name

        if language not in ["en", "zh"]:
            raise ValueError("language must be either 'en' or 'zh'")
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
            def area(s):
                return 1 - (1 - s ** float(r)) ** float(b)

            a, _ = integrate(area, 0.0, threshold)
            return a

        def false_negative_area(threshold: float, b: int, r: int):
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

    def run(self):
        ds = union_children(self.children, by_name=False)
        return self.minhash(ds)

    def minhash(self, ds):
        pdf = ds.to_pandas()
        if pdf.empty:
            return ds

        pdf = pdf[
            pdf[self.input_col_name].notna()
            & (pdf[self.input_col_name].astype(str).str.strip() != "")
        ].copy()
        num_doc = len(pdf)
        if num_doc == 0:
            return dataset_from_pandas(pdf)

        if "duplicate_count" not in pdf.columns:
            pdf["duplicate_count"] = 1.0

        pdf["uid"] = np.arange(len(pdf), dtype=np.int64)
        pdf["content"] = pdf[self.input_col_name].map(text_normalization)

        buckets = {}
        for row in pdf[["uid", "content"]].itertuples(index=False):
            hash_values = generate_hash_values(
                uid=int(row.uid),
                content=str(row.content),
                ngram_size=self.ngram_size,
                min_length=self.min_length,
                perm_a=self.perm_a,
                perm_b=self.perm_b,
                num_bands=self.num_bands,
                num_rows_per_band=self.num_rows_per_band,
                language=self.language,
            )
            for band, bucket_hash, uid in hash_values:
                buckets.setdefault((band, bucket_hash), []).append(uid)

        edges = set()
        for nodes in buckets.values():
            for edge in generate_edges(nodes):
                edges.add(tuple(edge))

        component_pairs = weakly_connected_component(edges, self.num_parallel)
        component_map = {int(vid): int(component) for vid, component in component_pairs}

        pdf["component"] = pdf["uid"].map(lambda uid: component_map.get(int(uid), int(uid)))
        # Keep historical behavior where the representative is always the minimum id.
        pdf["component"] = pdf[["uid", "component"]].min(axis=1)

        grouped = pdf.groupby("component")["duplicate_count"].sum()

        filtered = pdf[pdf["uid"] == pdf["component"]].copy()
        filtered["duplicate_count"] = filtered["component"].map(grouped).fillna(
            filtered["duplicate_count"]
        )

        dup_docs = num_doc - len(filtered)
        dup_rate = (dup_docs * 100.0 / num_doc) if num_doc else 0.0
        print(
            f"whole doc: {num_doc}, dup doc: {dup_docs}, duplicate rate: {dup_rate}%"
        )

        filtered = filtered.drop(columns=["uid", "component", "content"], errors="ignore")
        return dataset_from_pandas(filtered.reset_index(drop=True))
