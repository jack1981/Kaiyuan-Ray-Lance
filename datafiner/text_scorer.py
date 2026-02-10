from __future__ import annotations

import os
from abc import ABC, abstractmethod

import fasttext
import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from datafiner.base import PipelineNode
from datafiner.dataset_utils import map_batches_tuned, union_children
from datafiner.register import register


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

    return model_path


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

        def score_batch(batch: pd.DataFrame) -> pd.DataFrame:
            if not hasattr(score_batch, "model"):
                score_batch.model = fasttext.load_model(model_path)

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

        def score_batch(batch: pd.DataFrame) -> pd.DataFrame:
            if not hasattr(score_batch, "tokenizer"):
                resolved_model_path = _resolve_local_model_dir(model_path)
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
