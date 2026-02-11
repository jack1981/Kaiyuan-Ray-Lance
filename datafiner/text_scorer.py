"""Model-based scoring/filtering nodes for text datasets.

This module provides FastText and sequence-classifier scorers with helper
utilities for resolving local/remote model artifacts and caching them on disk.
These scorers represent dataset-specific quality metrics used for filtering and
mixing decisions in the PCMind-2.1 data recipe.
See also `script/prepare_local_sample.py` for sample model generation.
"""

from __future__ import annotations

import hashlib
import os
import shutil
import zipfile
from abc import ABC, abstractmethod
from pathlib import Path
from urllib.parse import urlparse

import fasttext
import pandas as pd
import pyarrow.fs as pafs
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from datafiner.base import PipelineNode
from datafiner.dataset_utils import map_batches_tuned, union_children
from datafiner.register import register


def _is_remote_uri(path: str) -> bool:
    """Check whether a path uses a remote object-store style scheme.

    Args:
        path: Candidate model path.

    Returns:
        True for known remote schemes, else False.

    Side effects:
        None.

    Assumptions:
        Remote schemes are limited to S3/GS/ABFS variants used in this project.
    """
    normalized = path.replace("\\", "/")
    return normalized.startswith(("s3://", "s3a://", "gs://", "abfs://"))


def _env_bool(name: str) -> bool | None:
    """Parse an optional boolean environment variable.

    Args:
        name: Environment variable name.

    Returns:
        `True`/`False` when set, else `None`.

    Side effects:
        Reads process environment variables.

    Assumptions:
        Truthy values are `1/true/yes` (case-insensitive).
    """
    value = os.getenv(name)
    if value is None:
        return None
    return str(value).strip().lower() in {"1", "true", "yes"}


def _model_cache_root() -> Path:
    """Return root directory used for downloaded/extracted model cache.

    Inputs/outputs:
        No inputs; returns cache root path.

    Side effects:
        Reads environment variable.

    Assumptions:
        Cache directory is writable by current process.
    """
    return Path(os.getenv("KAIYUAN_MODEL_CACHE_DIR", "/tmp/kaiyuan-ray-model-cache"))


def _normalize_s3_path(path: str) -> str:
    """Normalize `s3a://` paths to `s3://` for Arrow filesystem APIs.

    Args:
        path: Raw URI.

    Returns:
        Normalized URI.

    Side effects:
        None.

    Assumptions:
        Non-`s3a` paths should pass through unchanged.
    """
    if path.startswith("s3a://"):
        return "s3://" + path[len("s3a://") :]
    return path


def _split_s3_uri(path: str) -> tuple[str, str]:
    """Split an S3 URI into `(bucket, key)` components.

    Args:
        path: S3/S3A URI.

    Returns:
        Tuple containing bucket and object key.

    Side effects:
        None.

    Assumptions:
        URI points to a concrete object, not just a bucket root.
    """
    parsed = urlparse(_normalize_s3_path(path))
    if parsed.scheme != "s3" or not parsed.netloc:
        raise ValueError(f"Expected s3:// URI, got: {path}")
    key = parsed.path.lstrip("/")
    if not key:
        raise ValueError(f"Expected a concrete S3 object path, got: {path}")
    return parsed.netloc, key


def _build_s3_filesystem(storage_options: dict | None) -> pafs.S3FileSystem:
    """Construct Arrow S3 filesystem from storage options and env fallbacks.

    Args:
        storage_options: Optional Lance/Ray storage options.

    Returns:
        Configured `pyarrow.fs.S3FileSystem`.

    Side effects:
        Reads environment variables.

    Assumptions:
        Endpoint values may include or omit URL scheme.
    """
    options = dict(storage_options or {})

    endpoint = options.get("aws_endpoint") or os.getenv("AWS_ENDPOINT_URL") or os.getenv(
        "MINIO_ENDPOINT"
    )
    access_key = options.get("aws_access_key_id") or os.getenv("AWS_ACCESS_KEY_ID") or os.getenv(
        "MINIO_ACCESS_KEY"
    )
    secret_key = options.get("aws_secret_access_key") or os.getenv(
        "AWS_SECRET_ACCESS_KEY"
    ) or os.getenv("MINIO_SECRET_KEY")
    region = options.get("aws_region") or os.getenv("AWS_REGION") or os.getenv("MINIO_REGION")

    allow_http = options.get("aws_allow_http")
    if allow_http is None:
        allow_http = _env_bool("LANCE_AWS_ALLOW_HTTP")

    scheme = "http" if allow_http else "https"
    endpoint_override = None
    if endpoint:
        endpoint_text = str(endpoint)
        if "://" in endpoint_text:
            parsed = urlparse(endpoint_text)
            if parsed.scheme:
                scheme = parsed.scheme
            endpoint_override = parsed.netloc or parsed.path
        else:
            endpoint_override = endpoint_text

    # NOTE(readability): MinIO deployments in this repo rely on path-style URLs,
    # so virtual-host addressing is disabled for broad compatibility.
    kwargs = {"force_virtual_addressing": False, "scheme": scheme}
    if endpoint_override:
        kwargs["endpoint_override"] = endpoint_override
    if access_key:
        kwargs["access_key"] = access_key
    if secret_key:
        kwargs["secret_key"] = secret_key
    if region:
        kwargs["region"] = region

    return pafs.S3FileSystem(**kwargs)


def _download_remote_file(path: str, storage_options: dict | None) -> str:
    """Download a remote model artifact into deterministic local cache path.

    Args:
        path: Remote URI.
        storage_options: Optional storage options for S3 construction.

    Returns:
        Local filesystem path to cached file.

    Side effects:
        Performs network I/O and creates cache directories/files.

    Assumptions:
        Cache key is SHA256 of normalized URI and artifact immutability is
        acceptable for cache reuse.
    """
    normalized = _normalize_s3_path(path)
    parsed = urlparse(normalized)

    if parsed.scheme != "s3":
        fs, fs_path = pafs.FileSystem.from_uri(normalized)
        remote_path = str(fs_path).lstrip("/")
        suffix = Path(remote_path).suffix
    else:
        bucket, key = _split_s3_uri(path)
        fs = _build_s3_filesystem(storage_options)
        remote_path = f"{bucket}/{key}"
        suffix = Path(key).suffix

    cache_root = _model_cache_root() / "files"
    cache_root.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256(normalized.encode("utf-8")).hexdigest()
    target = cache_root / f"{digest}{suffix}"
    if target.is_file() and target.stat().st_size > 0:
        return str(target)

    tmp_target = target.with_suffix(target.suffix + ".tmp")
    if tmp_target.exists():
        tmp_target.unlink()

    # NOTE(readability): Download to `.tmp` then atomic replace to avoid partial
    # cache files being consumed by concurrent workers.
    with fs.open_input_stream(remote_path) as src, tmp_target.open("wb") as dst:
        shutil.copyfileobj(src, dst, length=8 * 1024 * 1024)
    tmp_target.replace(target)
    return str(target)


def _ensure_local_file(path: str, storage_options: dict | None) -> str:
    """Return local file path, downloading remote URIs when necessary.

    Args:
        path: Local path or remote URI.
        storage_options: Optional storage options for remote download.

    Returns:
        Local filesystem path.

    Side effects:
        May download remote file content.

    Assumptions:
        Local paths are already accessible without additional checks.
    """
    if not _is_remote_uri(path):
        return path
    return _download_remote_file(path, storage_options)


def _extract_zip_archive(path: str) -> str:
    """Extract a zip archive into content-addressed cache directory.

    Args:
        path: Local zip file path.

    Returns:
        Extracted directory path.

    Side effects:
        Creates/removes cache directories and extracts files.

    Assumptions:
        Archive content is safe to extract and marker file denotes completion.
    """
    archive_path = Path(path)
    if not archive_path.is_file():
        raise FileNotFoundError(f"Model archive does not exist: {path}")

    cache_root = _model_cache_root() / "dirs"
    cache_root.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256(str(archive_path).encode("utf-8")).hexdigest()
    extract_dir = cache_root / digest
    marker = extract_dir / ".ready"

    if marker.exists():
        return str(extract_dir)

    if extract_dir.exists():
        shutil.rmtree(extract_dir)
    extract_dir.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(archive_path, mode="r") as zf:
        zf.extractall(extract_dir)
    marker.touch()
    return str(extract_dir)


class TextScorer(PipelineNode, ABC):
    """Abstract text scorer that maps input text to numeric output columns.

    Inputs/outputs:
        Reads child dataset(s) and returns dataset with score column added or
        consumed by subclasses.

    Side effects:
        Subclasses load model artifacts and run distributed inference tasks.

    Assumptions:
        `input_col` is string-convertible and `output_col` is writable.
    """

    def __init__(
        self,
        runtime,
        model_path: str,
        output_col: str,
        input_col: str = "text",
        batch_size: int | None = None,
        concurrency: int | None = None,
        child_configs: list = None,
    ):
        """Initialize shared scorer settings.

        Args:
            runtime: Shared runtime config.
            model_path: Model file/directory/URI reference.
            output_col: Output score column name.
            input_col: Input text column name.
            batch_size: Optional per-stage batch size override.
            concurrency: Optional per-stage concurrency override.
            child_configs: Upstream node configs.

        Returns:
            None.

        Side effects:
            None during initialization.

        Assumptions:
            Concrete subclasses implement `score(ds)`.
        """
        super().__init__(runtime, child_configs)
        self.model_path = model_path
        self.output_col = output_col
        self.input_col = input_col
        self.batch_size = batch_size
        self.concurrency = concurrency

    @abstractmethod
    def score(self, ds):
        """Score a dataset and return transformed output dataset.

        Args:
            ds: Source dataset.

        Returns:
            Scored dataset.

        Side effects:
            Model inference and related I/O in subclass implementations.

        Assumptions:
            Implementations preserve row count/order unless documented otherwise.
        """
        pass

    def run(self):
        """Execute scorer against unioned child dataset output.

        Inputs/outputs:
            Reads child dataset(s) and returns scorer output dataset.

        Side effects:
            Executes child nodes and model-scoring tasks.

        Assumptions:
            Child datasets are union-compatible by position.
        """
        ds = union_children(self.children, by_name=False)
        return self.score(ds)


def _resolve_local_model_dir(model_path: str) -> str:
    """Resolve a local directory containing `config.json` for HF models.

    Args:
        model_path: Candidate path provided by config.

    Returns:
        Best matching directory path, or original input if unresolved.

    Side effects:
        Walks local filesystem directories.

    Assumptions:
        Presence of `config.json` identifies usable HF model directories.
    """
    normalized = model_path.rstrip("/")
    base_name = os.path.basename(normalized)

    candidates = [
        normalized,
        os.path.join(normalized, base_name),
        f"{normalized}.zip",
        os.path.join(f"{normalized}.zip", base_name),
        base_name,
        f"{base_name}.zip",
        os.path.join(f"{base_name}.zip", base_name),
    ]

    seen = set()
    for candidate in candidates:
        if not candidate or candidate in seen:
            continue
        seen.add(candidate)
        if os.path.isdir(candidate) and os.path.isfile(os.path.join(candidate, "config.json")):
            return candidate

    if os.path.isdir(normalized):
        for root, _dirs, files in os.walk(normalized):
            if "config.json" in files:
                return root

    return model_path


def _resolve_seq_model_path(model_path: str, storage_options: dict | None) -> str:
    """Resolve sequence-classifier model path from local/remote/zip references.

    Args:
        model_path: Model reference path or URI.
        storage_options: Optional storage options for remote download.

    Returns:
        Local model directory path suitable for HF loaders.

    Side effects:
        May download remote artifacts and extract zip archives.

    Assumptions:
        Zipped model artifacts contain an HF-compatible directory tree.
    """
    local_path = _ensure_local_file(model_path, storage_options)
    if str(local_path).endswith(".zip"):
        local_path = _extract_zip_archive(local_path)
    return _resolve_local_model_dir(local_path)


@register("FastTextScorer")
class FastTextScorer(TextScorer):
    """Score text using a supervised FastText model label probability.

    Inputs/outputs:
        Reads text column and adds float score column for `selected_label`.

    Side effects:
        Loads FastText model file (local or downloaded) inside worker tasks.

    Assumptions:
        Model exposes requested label among top-`num_labels` predictions.
        In PCMind-2.1 style recipes this can represent dataset-specific quality
        metrics (for example DCLM-style FastText quality scores).
    """

    def __init__(
        self,
        runtime,
        model_path: str,
        num_labels: int,
        selected_label: str,
        output_col: str,
        input_col: str = "text",
        batch_size: int | None = None,
        concurrency: int | None = None,
        child_configs: list = None,
    ):
        """Configure FastText scoring parameters.

        Args:
            runtime: Shared runtime config.
            model_path: FastText `.bin` model path/URI.
            num_labels: Number of labels requested per prediction.
            selected_label: Label whose probability is emitted.
            output_col: Score output column.
            input_col: Source text column.
            batch_size: Optional map-batches batch size override.
            concurrency: Optional map-batches concurrency override.
            child_configs: Upstream node configs.

        Returns:
            None.

        Side effects:
            None during initialization.

        Assumptions:
            Selected label probabilities default to 0 when label not returned.
        """
        super().__init__(
            runtime,
            model_path,
            output_col,
            input_col,
            batch_size,
            concurrency,
            child_configs,
        )
        self.num_labels = num_labels
        self.selected_label = selected_label

    def score(self, ds):
        """Run FastText scoring for each input row.

        Args:
            ds: Source dataset containing text column.

        Returns:
            Dataset with score column.

        Side effects:
            Loads model lazily in workers and runs inference.

        Assumptions:
            Empty/blank text rows produce score `0.0`.
        """
        model_path = self.model_path
        num_labels = self.num_labels
        selected_label = self.selected_label
        storage_options = self.runtime.storage_options

        def score_batch(batch: pd.DataFrame) -> pd.DataFrame:
            """Score one pandas batch with cached FastText model.

            Args:
                batch: Source pandas batch.

            Returns:
                Batch with `output_col` populated.

            Side effects:
                Loads model artifact on first call per worker.

            Assumptions:
                Function attribute cache persists across batch invocations.
            """
            if not hasattr(score_batch, "model"):
                # NOTE(readability): Worker-local lazy loading prevents repeated
                # model initialization cost for each micro-batch.
                resolved_model_path = _ensure_local_file(model_path, storage_options)
                score_batch.model = fasttext.load_model(resolved_model_path)

            out = batch.copy()
            texts = out[self.input_col].fillna("").astype(str).tolist()

            def _score_text(text: str) -> float:
                """Score one cleaned text string for the selected label.

                Args:
                    text: Raw text value.

                Returns:
                    Probability score for `selected_label`.

                Side effects:
                    None.

                Assumptions:
                    Text is normalized to lowercase single-line form before
                    prediction.
                """
                clean_text = str(text).replace("\n", " ").replace("\r", " ").strip().lower()
                if not clean_text:
                    return 0.0
                labels, probs = score_batch.model.predict(clean_text, num_labels)
                labels = list(labels)
                probs = list(probs)
                if selected_label in labels:
                    return float(probs[labels.index(selected_label)])
                return 0.0

            out[self.output_col] = [_score_text(text) for text in texts]
            return out

        return map_batches_tuned(
            ds,
            self.runtime,
            score_batch,
            batch_format="pandas",
            batch_size=self.batch_size,
            concurrency=self.concurrency,
        )


@register("FastTextFilter")
class FastTextFilter(FastTextScorer):
    """FastText scorer variant that keeps rows above a threshold.

    Inputs/outputs:
        Scores rows, filters by threshold, and drops temporary score column.

    Side effects:
        Inherits FastText model loading/inference behavior.

    Assumptions:
        Filter threshold is applied on numeric-cast temporary score values.
    """

    def __init__(
        self,
        runtime,
        model_path: str,
        num_labels: int,
        selected_label: str,
        input_col: str = "text",
        temp_col: str = "filter_score",
        child_configs: list = None,
        threshold: float = 0.5,
        batch_size: int | None = None,
        concurrency: int | None = None,
    ):
        """Configure FastText filtering threshold and temporary score column.

        Args:
            runtime: Shared runtime config.
            model_path: FastText model path/URI.
            num_labels: Number of labels requested from model.
            selected_label: Label used for score selection.
            input_col: Source text column.
            temp_col: Temporary score column used during filtering.
            child_configs: Upstream node configs.
            threshold: Keep rows strictly above this score.
            batch_size: Optional batch size override.
            concurrency: Optional concurrency override.

        Returns:
            None.

        Side effects:
            None during initialization.

        Assumptions:
            Temporary score column can be safely dropped in output.
        """
        super().__init__(
            runtime,
            model_path,
            num_labels,
            selected_label,
            temp_col,
            input_col,
            batch_size,
            concurrency,
            child_configs,
        )
        self.threshold = threshold
        self.temp_col = temp_col

    def run(self):
        """Execute score-then-filter workflow on child dataset output.

        Inputs/outputs:
            Reads child dataset(s) and returns filtered dataset.

        Side effects:
            Runs model inference and filter transform.

        Assumptions:
            Child datasets are union-compatible by position.
        """
        ds = union_children(self.children, by_name=False)
        return self.filter(ds)

    def filter(self, ds):
        """Filter scored rows using configured threshold.

        Args:
            ds: Source dataset.

        Returns:
            Filtered dataset without temporary score column.

        Side effects:
            Runs scoring and batch filtering transforms.

        Assumptions:
            Threshold comparison uses strict greater-than semantics.
        """
        scored = self.score(ds)

        def apply_filter(batch: pd.DataFrame) -> pd.DataFrame:
            """Filter one batch by temporary score column threshold.

            Args:
                batch: Scored pandas batch.

            Returns:
                Batch containing only rows above threshold.

            Side effects:
                None.

            Assumptions:
                Non-numeric temp values are coerced to NaN and dropped.
            """
            out = batch[pd.to_numeric(batch[self.temp_col], errors="coerce") > self.threshold].copy()
            return out.drop(columns=[self.temp_col], errors="ignore")

        return map_batches_tuned(
            scored,
            self.runtime,
            apply_filter,
            batch_format="pandas",
            batch_size=self.batch_size,
            concurrency=self.concurrency,
        )


@register("SeqClassifierScorer")
class SeqClassifierScorer(TextScorer):
    """Score text with Hugging Face sequence classification logits.

    Inputs/outputs:
        Reads text column and emits one selected-logit score per row.

    Side effects:
        Resolves model paths (including remote/zip), loads tokenizer/model in
        workers, and runs PyTorch inference.

    Assumptions:
        `selected_index` references logits index; out-of-range yields 0.
    """

    def __init__(
        self,
        runtime,
        model_path: str,
        output_col: str,
        selected_index: int = 0,
        input_col: str = "text",
        batch_size: int | None = None,
        concurrency: int | None = None,
        child_configs: list = None,
    ):
        """Configure sequence-classifier scoring parameters.

        Args:
            runtime: Shared runtime config.
            model_path: Local/remote model path.
            output_col: Output score column.
            selected_index: Logit index to export as score.
            input_col: Source text column.
            batch_size: Optional batch size override.
            concurrency: Optional concurrency override.
            child_configs: Upstream node configs.

        Returns:
            None.

        Side effects:
            None during initialization.

        Assumptions:
            Model and tokenizer are HF-compatible at resolved model path.
        """
        super().__init__(
            runtime,
            model_path,
            output_col,
            input_col,
            batch_size,
            concurrency,
            child_configs,
        )
        self.selected_index = selected_index

    def score(self, ds):
        """Run HF sequence-classifier inference and emit selected logits.

        Args:
            ds: Source dataset.

        Returns:
            Dataset with score column.

        Side effects:
            Loads model/tokenizer lazily in workers and runs PyTorch inference.

        Assumptions:
            Text is truncated to 512 tokens with padding enabled.
        """
        model_path = self.model_path
        selected_index = self.selected_index
        storage_options = self.runtime.storage_options

        def score_batch(batch: pd.DataFrame) -> pd.DataFrame:
            """Score one pandas batch with cached tokenizer/model pair.

            Args:
                batch: Source pandas batch.

            Returns:
                Batch with selected-logit scores.

            Side effects:
                Performs lazy model load and CPU inference.

            Assumptions:
                Empty text rows keep default score `0.0`.
            """
            if not hasattr(score_batch, "tokenizer"):
                # NOTE(readability): Resolve and load once per worker to avoid
                # repeated remote download/extraction and model init overhead.
                resolved_model_path = _resolve_seq_model_path(model_path, storage_options)
                score_batch.tokenizer = AutoTokenizer.from_pretrained(resolved_model_path)
                score_batch.model = AutoModelForSequenceClassification.from_pretrained(
                    resolved_model_path
                )
                score_batch.model.eval()

            out = batch.copy()
            texts = [str(x).replace("\n", " ").replace("\r", " ").strip() for x in out[self.input_col].fillna("").tolist()]
            scores = [0.0] * len(texts)

            valid_indices = [idx for idx, text in enumerate(texts) if text]
            if valid_indices:
                valid_texts = [texts[idx] for idx in valid_indices]
                input_ids = score_batch.tokenizer(
                    valid_texts,
                    return_tensors="pt",
                    max_length=512,
                    padding=True,
                    truncation=True,
                )
                with torch.no_grad():
                    outputs = score_batch.model(**input_ids)
                    logits = outputs.logits.detach().cpu().float().numpy()

                if logits.ndim == 1:
                    logits = logits.reshape(1, -1)

                for pos, idx in enumerate(valid_indices):
                    row_logits = logits[pos]
                    if selected_index < len(row_logits):
                        scores[idx] = float(row_logits[selected_index])
                    else:
                        scores[idx] = 0.0

            out[self.output_col] = scores
            return out

        return map_batches_tuned(
            ds,
            self.runtime,
            score_batch,
            batch_format="pandas",
            batch_size=self.batch_size,
            concurrency=self.concurrency,
        )
