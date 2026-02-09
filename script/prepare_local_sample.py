#!/usr/bin/env python3
import argparse
import os
from typing import List

import pyarrow as pa
import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download, list_repo_files


def _pick_source_file(repo_id: str) -> str:
    files: List[str] = list_repo_files(repo_id=repo_id, repo_type="dataset")

    parquet_files = [f for f in files if f.endswith(".parquet")]
    if parquet_files:
        parquet_files.sort()
        return parquet_files[0]

    raise RuntimeError(
        f"No parquet files found in dataset repository: {repo_id}. "
        "Set --source-file manually once you identify a valid file."
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create a small local Parquet sample from a Hugging Face dataset file"
    )
    parser.add_argument("--dataset", default="thu-pacman/PCMind-2.1-Kaiyuan-2B")
    parser.add_argument(
        "--source-file",
        default="",
        help="Optional dataset file path inside HF repo. Defaults to first parquet file.",
    )
    parser.add_argument("--rows", type=int, default=200)
    parser.add_argument("--output", default="/data/sample/pcmind_kaiyuan_2b_sample.parquet")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    source_file = args.source_file.strip() or _pick_source_file(args.dataset)
    local_file = hf_hub_download(
        repo_id=args.dataset,
        filename=source_file,
        repo_type="dataset",
    )

    table = pq.read_table(local_file)
    if table.num_rows == 0:
        raise RuntimeError(f"Downloaded source file has 0 rows: {source_file}")

    sample_rows = min(args.rows, table.num_rows)
    sample = table.slice(0, sample_rows)

    pq.write_table(sample, args.output)

    print(f"Dataset: {args.dataset}")
    print(f"Source file: {source_file}")
    print(f"Wrote {sample_rows} rows to {args.output}")


if __name__ == "__main__":
    main()
