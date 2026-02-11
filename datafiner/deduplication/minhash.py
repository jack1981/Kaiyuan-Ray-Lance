"""MinHash-based near-duplicate clustering and collapse node.

This module tokenizes normalized text into n-grams, computes MinHash signatures,
builds candidate duplicate edges, clusters by connected components, and keeps
one representative row per cluster with aggregated duplicate counts.
This corresponds to the report's recommendation to deduplicate before quality
quantile benchmarking and curriculum construction.
See also `datafiner/deduplication/text_normalization.py`.
"""

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
NON_ALPHA = re.compile(r"\W", re.UNICODE)


def ngrams(sequence: List[Text], n: int, min_length: int = 5):
    """Yield n-grams from a token sequence with short-sequence handling.

    Args:
        sequence: Token list.
        n: N-gram size.
        min_length: Minimum sequence length to emit any grams.

    Returns:
        Iterable of n-gram tuples (or empty list for short sequences).

    Side effects:
        None.

    Assumptions:
        Sequences shorter than `n` but above `min_length` produce one full tuple.
    """
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
    """Hash bytes into a 64-bit integer using SHA-1 prefix.

    Args:
        data: Byte payload.

    Returns:
        Unsigned 64-bit hash integer.

    Side effects:
        None.

    Assumptions:
        First 8 bytes of SHA-1 digest provide sufficient spread for MinHash use.
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
    """Generate MinHash LSH band hashes for one document.

    Args:
        uid: Unique document id.
        content: Normalized text content.
        ngram_size: N-gram size used for shingles.
        min_length: Minimum token length to emit shingles.
        perm_a: MinHash permutation multiplier vector.
        perm_b: MinHash permutation offset vector.
        num_bands: Number of LSH bands.
        num_rows_per_band: Signature rows per band.
        language: Tokenization mode (`en` or `zh`).

    Returns:
        List of `(band_index, band_hash_bytes, uid)` tuples.

    Side effects:
        Uses jieba tokenization for Chinese mode.

    Assumptions:
        Input text is pre-normalized and non-empty for meaningful signatures.
    """
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
    """Generate star edges connecting all nodes to minimum representative.

    Args:
        nodes: Node ids colliding in one LSH bucket.

    Returns:
        Edge list suitable for connected-component clustering.

    Side effects:
        None.

    Assumptions:
        Star topology is sufficient because WCC transitivity recovers full group.
    """
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
        """Configure MinHash deduplication parameters.

        Args:
            runtime: Shared runtime config.
            num_permutations: Initial signature permutation count.
            threshold: Target Jaccard-like near-duplicate threshold.
            num_parallel: Compatibility arg for downstream WCC API.
            language: Tokenization language (`en` or `zh`).
            child_configs: Upstream node configs.
            input_col_name: Text column to deduplicate.
            seed: RNG seed for permutation generation.
            ngram_size: Token n-gram shingle size.
            min_length: Minimum token count for shingle generation.

        Returns:
            None.

        Side effects:
            Computes optimal LSH band/row layout and random permutation vectors.

        Assumptions:
            Language is limited to English/Chinese tokenization paths.
        """
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
        """Search LSH band/row layout minimizing weighted FP/FN area.

        Args:
            false_positive_weight: Weight for false-positive area term.
            false_negative_weight: Weight for false-negative area term.

        Returns:
            None; updates `num_bands`, `num_rows_per_band`, and effective
            `num_permutations` in place.

        Side effects:
            Runs numerical integrations across candidate parameter grid.

        Assumptions:
            Brute-force scan over divisors is acceptable at init-time scale.
        """
        def false_positive_area(threshold: float, b: int, r: int):
            """Integrate probability mass where dissimilar pairs collide.

            Args:
                threshold: Similarity threshold.
                b: Number of bands.
                r: Rows per band.

            Returns:
                Approximate false-positive area under LSH collision curve.

            Side effects:
                None.

            Assumptions:
                Standard LSH S-curve approximation applies.
            """
            def area(s):
                """LSH collision probability curve for similarity `s`.

                Args:
                    s: Similarity value in `[0, 1]`.

                Returns:
                    Collision probability estimate.

                Side effects:
                    None.

                Assumptions:
                    Standard banding formula `(1 - (1 - s^r)^b)` applies.
                """
                return 1 - (1 - s ** float(r)) ** float(b)

            a, _ = integrate(area, 0.0, threshold)
            return a

        def false_negative_area(threshold: float, b: int, r: int):
            """Integrate probability mass where similar pairs fail to collide.

            Args:
                threshold: Similarity threshold.
                b: Number of bands.
                r: Rows per band.

            Returns:
                Approximate false-negative area under collision complement.

            Side effects:
                None.

            Assumptions:
                Uses same LSH S-curve model as false-positive integral.
            """
            def area(s):
                """Complement collision probability for false-negative region.

                Args:
                    s: Similarity value in `[0, 1]`.

                Returns:
                    Non-collision probability estimate for similar pairs.

                Side effects:
                    None.

                Assumptions:
                    Uses complement of standard LSH collision formula.
                """
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
        """Run MinHash deduplication on child dataset output.

        Inputs/outputs:
            Reads child dataset(s) and returns deduplicated representative rows.

        Side effects:
            Delegates to pandas-materializing dedup pipeline.

        Assumptions:
            Children are union-compatible by position.
        """
        ds = union_children(self.children, by_name=False)
        return self.minhash(ds)

    def minhash(self, ds):
        """Compute MinHash buckets, cluster duplicates, and keep representatives.

        Args:
            ds: Source dataset containing text column.

        Returns:
            Deduplicated dataset with updated `duplicate_count`.

        Side effects:
            Materializes full dataset to pandas, runs hashing/tokenization, and
            prints duplicate-rate summary.

        Assumptions:
            Representative row per cluster is chosen by minimum generated uid.
        """
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

        # NOTE(readability): Bucket keys are `(band, hash)` so candidate edges
        # are only created for documents sharing at least one LSH band.
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
        # NOTE(readability): Keep historical behavior where the representative is
        # always the minimum id to preserve deterministic row selection.
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
