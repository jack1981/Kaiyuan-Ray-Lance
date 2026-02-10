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
    normalized = path.replace("\\", "/")
    return normalized.startswith(("s3://", "s3a://", "gs://", "abfs://"))


def _env_bool(name: str) -> bool | None:
    value = os.getenv(name)
    if value is None:
        return None
    return str(value).strip().lower() in {"1", "true", "yes"}


def _model_cache_root() -> Path:
    return Path(os.getenv("KAIYUAN_MODEL_CACHE_DIR", "/tmp/kaiyuan-ray-model-cache"))


def _normalize_s3_path(path: str) -> str:
    if path.startswith("s3a://"):
        return "s3://" + path[len("s3a://") :]
    return path


def _split_s3_uri(path: str) -> tuple[str, str]:
    parsed = urlparse(_normalize_s3_path(path))
    if parsed.scheme != "s3" or not parsed.netloc:
        raise ValueError(f"Expected s3:// URI, got: {path}")
    key = parsed.path.lstrip("/")
    if not key:
        raise ValueError(f"Expected a concrete S3 object path, got: {path}")
    return parsed.netloc, key


def _build_s3_filesystem(storage_options: dict | None) -> pafs.S3FileSystem:
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

    with fs.open_input_stream(remote_path) as src, tmp_target.open("wb") as dst:
        shutil.copyfileobj(src, dst, length=8 * 1024 * 1024)
    tmp_target.replace(target)
    return str(target)


def _ensure_local_file(path: str, storage_options: dict | None) -> str:
    if not _is_remote_uri(path):
        return path
    return _download_remote_file(path, storage_options)


def _extract_zip_archive(path: str) -> str:
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
        super().__init__(runtime, child_configs)
        self.model_path = model_path
        self.output_col = output_col
        self.input_col = input_col
        self.batch_size = batch_size
        self.concurrency = concurrency

    @abstractmethod
    def score(self, ds):
        pass

    def run(self):
        ds = union_children(self.children, by_name=False)
        return self.score(ds)


def _resolve_local_model_dir(model_path: str) -> str:
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
    local_path = _ensure_local_file(model_path, storage_options)
    if str(local_path).endswith(".zip"):
        local_path = _extract_zip_archive(local_path)
    return _resolve_local_model_dir(local_path)


@register("FastTextScorer")
class FastTextScorer(TextScorer):
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
        model_path = self.model_path
        num_labels = self.num_labels
        selected_label = self.selected_label
        storage_options = self.runtime.storage_options

        def score_batch(batch: pd.DataFrame) -> pd.DataFrame:
            if not hasattr(score_batch, "model"):
                resolved_model_path = _ensure_local_file(model_path, storage_options)
                score_batch.model = fasttext.load_model(resolved_model_path)

            out = batch.copy()
            texts = out[self.input_col].fillna("").astype(str).tolist()

            def _score_text(text: str) -> float:
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
        ds = union_children(self.children, by_name=False)
        return self.filter(ds)

    def filter(self, ds):
        scored = self.score(ds)

        def apply_filter(batch: pd.DataFrame) -> pd.DataFrame:
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
        model_path = self.model_path
        selected_index = self.selected_index
        storage_options = self.runtime.storage_options

        def score_batch(batch: pd.DataFrame) -> pd.DataFrame:
            if not hasattr(score_batch, "tokenizer"):
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
