from abc import ABC, abstractmethod
import fasttext
import os
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    LlamaTokenizerFast,
)
import torch
import re
import unicodedata
from typing import Tuple

from pyspark import SparkFiles
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql import functions as F
from pyspark.sql.types import FloatType
from datafiner.base import PipelineNode
from datafiner.register import register


class TextScorer(PipelineNode, ABC):
    def __init__(
        self,
        spark: SparkSession,
        model_path: str,
        output_col: str,
        input_col: str = "text",
        child_configs: list = None,
    ):
        super().__init__(spark, child_configs)
        self.model_path = model_path
        self.output_col = output_col
        self.input_col = input_col

    @abstractmethod
    def score(self, df: DataFrame) -> DataFrame:
        pass

    def run(self):
        df_union = self.children[0].run()
        if len(self.children) > 1:
            for child in self.children[1:]:
                df = child.run()
                df_union = df_union.union(df)
        return self.score(df_union)


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

    try:
        spark_files_root = SparkFiles.getRootDirectory()
        candidates.extend(
            [
                os.path.join(spark_files_root, normalized),
                os.path.join(spark_files_root, base_name),
                os.path.join(spark_files_root, f"{base_name}.zip"),
                os.path.join(spark_files_root, f"{base_name}.zip", base_name),
            ]
        )
    except Exception:
        pass

    seen = set()
    for candidate in candidates:
        if not candidate:
            continue
        if candidate in seen:
            continue
        seen.add(candidate)
        if os.path.isdir(candidate):
            config_path = os.path.join(candidate, "config.json")
            if os.path.isfile(config_path):
                return candidate

    return model_path


@register("FastTextScorer")
class FastTextScorer(TextScorer):
    def __init__(
        self,
        spark: SparkSession,
        model_path: str,
        num_labels: int,
        selected_label: str,
        output_col: str,
        input_col: str = "text",
        child_configs: list = None,
    ):
        super().__init__(
            spark,
            model_path,
            output_col,
            input_col,
            child_configs,
        )
        self.num_labels = num_labels
        self.selected_label = selected_label

    def score(self, df: DataFrame) -> DataFrame:
        model_path = self.model_path
        num_labels = self.num_labels
        selected_label = self.selected_label

        def score_udf(text):
            if not hasattr(score_udf, "model"):
                score_udf.model = fasttext.load_model(model_path)
            clean_text = text.replace("\n", " ").replace("\r", " ").strip().lower()
            if not clean_text:
                return 0.0
            labels, probs = score_udf.model.predict(clean_text, num_labels)
            if selected_label in labels:
                label_index = labels.index(selected_label)
                return float(probs[label_index])
            else:
                return 0.0

        func = F.udf(score_udf, FloatType())
        result = df.withColumn(self.output_col, func(F.col(self.input_col)))
        return result


@register("FastTextFilter")
class FastTextFilter(FastTextScorer):
    def __init__(
        self,
        spark: SparkSession,
        model_path: str,
        num_labels: int,
        selected_label: str,
        input_col: str = "text",
        temp_col: str = "filter_score",
        child_configs: list = None,
        threshold: float = 0.5,
    ):
        super().__init__(
            spark,
            model_path,
            num_labels,
            selected_label,
            temp_col,
            input_col,
            child_configs,
        )
        self.model_path = model_path
        self.num_labels = num_labels
        self.selected_label = selected_label
        self.input_col = input_col
        self.threshold = threshold
        self.temp_col = temp_col

    def run(self):
        df = self.children[0].run()
        if len(self.children) > 1:
            for child in self.children[1:]:
                df = df.union(child.run())
        return self.filter(df)

    def filter(self, df: DataFrame) -> DataFrame:
        df = self.score(df)
        df = df.filter(F.col(self.temp_col) > self.threshold)
        df = df.drop(self.temp_col)
        return df


@register("SeqClassifierScorer")
class SeqClassifierScorer(TextScorer):
    def __init__(
        self,
        spark: SparkSession,
        model_path: str,
        output_col: str,
        selected_index: int = 0,
        input_col: str = "text",
        child_configs: list = None,
    ):
        super().__init__(
            spark,
            model_path,
            output_col,
            input_col,
            child_configs,
        )
        self.selected_index = selected_index

    def score(self, df: DataFrame) -> DataFrame:
        model_path = self.model_path
        selected_index = self.selected_index

        def score_udf(text):
            if not hasattr(score_udf, "tokenizer"):
                resolved_model_path = _resolve_local_model_dir(model_path)
                score_udf.tokenizer = AutoTokenizer.from_pretrained(
                    resolved_model_path
                )
                score_udf.model = AutoModelForSequenceClassification.from_pretrained(
                    resolved_model_path
                )
                score_udf.model.eval()
            clean_text = text.replace("\n", " ").replace("\r", " ").strip()
            if not clean_text:
                return 0.0

            input_id = score_udf.tokenizer(
                clean_text,
                return_tensors="pt",
                max_length=512,
                padding=True,
                truncation=True,
            )
            with torch.no_grad():
                outputs = score_udf.model(**input_id)
                logits = outputs.logits.detach().cpu().float().numpy()

            if logits.ndim == 2:
                return float(logits[0][selected_index])
            if logits.ndim == 1:
                return float(logits[selected_index])
            return float(logits.reshape(-1)[selected_index])

        func = F.udf(score_udf, FloatType())
        result = df.withColumn(self.output_col, func(F.col(self.input_col)))
        return result
