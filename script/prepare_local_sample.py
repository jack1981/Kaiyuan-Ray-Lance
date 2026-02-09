#!/usr/bin/env python3
import argparse
import json
import os
import signal
import shutil
from pathlib import Path
from typing import Iterable, List

import fasttext
import pyarrow as pa
import pyarrow.parquet as pq
from huggingface_hub import HfApi, hf_hub_download, list_repo_files
from pyspark.sql import SparkSession
from transformers import BertConfig, BertForSequenceClassification, BertTokenizerFast


_LANCE_SPARK_SESSION = None


def _as_lance_identifier(path: str) -> str:
    if path.startswith("lance."):
        return path
    return f"lance.`{path.replace('`', '``')}`"


def _get_lance_spark_session() -> SparkSession:
    global _LANCE_SPARK_SESSION
    if _LANCE_SPARK_SESSION is not None:
        return _LANCE_SPARK_SESSION

    builder = (
        SparkSession.builder.appName("prepare_local_sample_lance")
        .master("local[*]")
        .config("spark.sql.catalog.lance", "com.lancedb.lance.spark.LanceCatalog")
        .config("spark.sql.extensions", "com.lancedb.lance.spark.extensions.LanceSparkSessionExtensions")
        .config("spark.sql.shuffle.partitions", "8")
    )

    spark_home = os.environ.get("SPARK_HOME")
    lance_jar = None
    if spark_home:
        for candidate in sorted(Path(spark_home).glob("jars/lance-spark-bundle*.jar")):
            lance_jar = str(candidate)
            break

    if lance_jar:
        builder = builder.config("spark.jars", lance_jar)
    else:
        lance_pkg = os.environ.get(
            "LANCE_SPARK_PACKAGE", "com.lancedb:lance-spark-bundle-3.5_2.12:0.0.15"
        )
        builder = builder.config("spark.jars.packages", lance_pkg)

    _LANCE_SPARK_SESSION = builder.getOrCreate()
    return _LANCE_SPARK_SESSION


def _pick_first_parquet(repo_id: str, max_size_bytes: int | None = None) -> str:
    api = HfApi()
    try:
        entries = list(api.list_repo_tree(repo_id=repo_id, repo_type="dataset", recursive=True))
        parquet_entries = [e for e in entries if getattr(e, "path", "").endswith(".parquet")]
        if not parquet_entries:
            raise RuntimeError(f"No parquet files found in dataset repo: {repo_id}")

        if max_size_bytes is not None:
            small = [e for e in parquet_entries if getattr(e, "size", None) is not None and e.size <= max_size_bytes]
            if small:
                return sorted(small, key=lambda x: (x.size, x.path))[0].path

        with_size = [e for e in parquet_entries if getattr(e, "size", None) is not None]
        if with_size:
            return sorted(with_size, key=lambda x: (x.size, x.path))[0].path
        return sorted(parquet_entries, key=lambda x: x.path)[0].path
    except Exception:
        files = list_repo_files(repo_id=repo_id, repo_type="dataset")
        parquet_files = sorted([f for f in files if f.endswith(".parquet")])
        if not parquet_files:
            raise RuntimeError(f"No parquet files found in dataset repo: {repo_id}")
        return parquet_files[0]


def _load_texts_from_hf(dataset: str, rows: int, max_size_bytes: int | None = None) -> List[str]:
    source_file = _pick_first_parquet(dataset, max_size_bytes=max_size_bytes)
    local_file = hf_hub_download(repo_id=dataset, filename=source_file, repo_type="dataset")
    table = pq.read_table(local_file)

    if table.num_rows == 0:
        raise RuntimeError(f"Dataset parquet has zero rows: {source_file}")

    col_names = set(table.schema.names)
    text_col = None
    for candidate in ["text", "content", "raw_content", "document", "body"]:
        if candidate in col_names:
            text_col = candidate
            break

    if text_col is None:
        for field in table.schema:
            if pa.types.is_string(field.type) or pa.types.is_large_string(field.type):
                text_col = field.name
                break

    if text_col is None:
        raise RuntimeError("No string column found in downloaded parquet.")

    selected = table.slice(0, min(rows, table.num_rows))
    col = selected.column(text_col)

    texts = []
    for value in col.to_pylist():
        if value is None:
            continue
        s = str(value).strip()
        if s:
            texts.append(s)

    if not texts:
        raise RuntimeError("No valid text rows found in downloaded parquet.")

    print(f"Using HF dataset column '{text_col}' from {source_file}")
    return texts[:rows]


def _fallback_texts(rows: int) -> List[str]:
    base = [
        "Open-source language model training requires scalable data pipelines.",
        "Spark local mode is useful for rapid preprocessing validation.",
        "High quality text filtering improves downstream model performance.",
        "Deduplication helps reduce memorization and near-copy artifacts.",
        "Tokenization and data mixing are core stages in corpus preparation.",
    ]
    out = []
    for i in range(rows):
        out.append(f"{base[i % len(base)]} sample_id={i}")
    return out


def _write_lance(path: Path, rows: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.is_dir():
            shutil.rmtree(path)
        else:
            path.unlink()
    table = pa.Table.from_pylist(list(rows))
    spark = _get_lance_spark_session()
    spark.createDataFrame(table.to_pandas()).writeTo(_as_lance_identifier(str(path))).using(
        "lance"
    ).createOrReplace()


def _build_scored_rows(texts: List[str]) -> List[dict]:
    scored = []
    for i, text in enumerate(texts):
        scored.append(
            {
                "id": i,
                "text": text,
                "score": float((i % 100) / 100.0),
                "fasttext_score": float(((i * 7) % 100) / 100.0),
                "duplicate_count": int(1 + (i % 4)),
            }
        )
    return scored


def _write_fasttext_train_file(path: Path, labeled_rows: List[tuple[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for label, text in labeled_rows:
            clean = text.replace("\n", " ").replace("\r", " ").strip()
            if clean:
                f.write(f"{label} {clean}\n")


def _train_fasttext_model(train_path: Path, model_prefix: Path) -> Path:
    model = fasttext.train_supervised(
        input=str(train_path),
        epoch=25,
        lr=0.8,
        wordNgrams=2,
        dim=32,
        minCount=1,
        verbose=0,
    )
    model.save_model(str(model_prefix) + ".bin")
    return Path(str(model_prefix) + ".bin")


def _prepare_tiny_seq_classifier(model_dir: Path) -> Path:
    seq_dir = model_dir / "tiny_seq_classifier"
    seq_dir.mkdir(parents=True, exist_ok=True)

    vocab = [
        "[PAD]",
        "[UNK]",
        "[CLS]",
        "[SEP]",
        "[MASK]",
        "open",
        "source",
        "spark",
        "data",
        "model",
        "pipeline",
        "quality",
        "text",
        "token",
        "train",
        "sample",
        "local",
        "score",
        ".",
        ",",
        "-",
    ]
    vocab_path = seq_dir / "vocab.txt"
    with vocab_path.open("w", encoding="utf-8") as f:
        f.write("\\n".join(vocab) + "\\n")

    tokenizer = BertTokenizerFast(vocab_file=str(vocab_path), do_lower_case=True)
    tokenizer.save_pretrained(str(seq_dir))

    config = BertConfig(
        vocab_size=len(vocab),
        hidden_size=64,
        num_hidden_layers=2,
        num_attention_heads=2,
        intermediate_size=128,
        num_labels=2,
    )
    model = BertForSequenceClassification(config)
    model.save_pretrained(str(seq_dir))
    return seq_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare local sample data/models for all example YAMLs")
    parser.add_argument("--dataset", default="thu-pacman/PCMind-2.1-Kaiyuan-2B")
    parser.add_argument("--rows", type=int, default=200)
    parser.add_argument("--data-root", default="/data")
    parser.add_argument(
        "--source-mode",
        choices=["auto", "hf", "synthetic"],
        default="auto",
        help="auto: try HF then fallback, hf: strict HF, synthetic: no network",
    )
    parser.add_argument(
        "--hf-timeout-seconds",
        type=int,
        default=30,
        help="Timeout for HF metadata/download in auto/hf modes.",
    )
    parser.add_argument(
        "--max-hf-parquet-mb",
        type=int,
        default=256,
        help="In auto mode, prefer parquet files at or below this size.",
    )
    args = parser.parse_args()

    data_root = Path(args.data_root)
    sample_dir = data_root / "sample"
    output_dir = data_root / "output"
    model_dir = data_root / "models"

    sample_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)

    texts: List[str]
    source = "synthetic"

    def _alarm_handler(signum, frame):
        raise TimeoutError(f"HF fetch timed out after {args.hf_timeout_seconds}s")

    if args.source_mode in ("auto", "hf"):
        old_handler = signal.signal(signal.SIGALRM, _alarm_handler)
        signal.alarm(args.hf_timeout_seconds)
        try:
            max_size_bytes = None
            if args.source_mode == "auto":
                max_size_bytes = args.max_hf_parquet_mb * 1024 * 1024
            texts = _load_texts_from_hf(args.dataset, args.rows, max_size_bytes=max_size_bytes)
            source = "huggingface"
        except Exception as exc:
            if args.source_mode == "hf":
                raise
            print(f"HF unavailable, using synthetic fallback: {exc}")
            texts = _fallback_texts(args.rows)
            source = "synthetic_fallback"
        finally:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, old_handler)
    else:
        texts = _fallback_texts(args.rows)
        source = "synthetic"

    scored_rows = _build_scored_rows(texts)

    base_path = sample_dir / "pcmind_kaiyuan_2b_sample.lance"
    scored_path = sample_dir / "scored_input.lance"
    fineweb_path = sample_dir / "fineweb_chinese.lance"
    dclm_subset_path = sample_dir / "dclm_subset.lance"
    dedup_subset_path = sample_dir / "dclm_subset_dedup.lance"

    _write_lance(base_path, ({"text": t} for t in texts))
    _write_lance(scored_path, scored_rows)
    _write_lance(fineweb_path, scored_rows)
    _write_lance(dclm_subset_path, scored_rows)
    _write_lance(dedup_subset_path, scored_rows)

    hq_train_path = model_dir / "fasttext_hq_train.txt"
    mmlu_train_path = model_dir / "fasttext_mmlu_train.txt"

    hq_rows = []
    mmlu_rows = []
    for i, row in enumerate(scored_rows):
        text = row["text"]
        hq_rows.append(("__label__hq" if i % 2 == 0 else "__label__lq", text))
        mmlu_rows.append(("__label__mmlu_" if i % 2 == 0 else "__label__other", text))

    _write_fasttext_train_file(hq_train_path, hq_rows)
    _write_fasttext_train_file(mmlu_train_path, mmlu_rows)

    fasttext_hq_model = _train_fasttext_model(hq_train_path, model_dir / "fasttext_hq")
    fasttext_mmlu_model = _train_fasttext_model(mmlu_train_path, model_dir / "fasttext_mmlu")
    tiny_seq_classifier_dir = _prepare_tiny_seq_classifier(model_dir)

    manifest = {
        "source": source,
        "rows": len(texts),
        "dataset": args.dataset,
        "paths": {
            "base": str(base_path),
            "scored": str(scored_path),
            "fineweb_chinese": str(fineweb_path),
            "dclm_subset": str(dclm_subset_path),
            "dclm_subset_dedup": str(dedup_subset_path),
            "fasttext_hq_model": str(fasttext_hq_model),
            "fasttext_mmlu_model": str(fasttext_mmlu_model),
            "tiny_seq_classifier_model": str(tiny_seq_classifier_dir),
        },
    }

    manifest_path = sample_dir / "manifest.json"
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    global _LANCE_SPARK_SESSION
    if _LANCE_SPARK_SESSION is not None:
        _LANCE_SPARK_SESSION.stop()
        _LANCE_SPARK_SESSION = None

    print("Prepared local assets for all examples:")
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
