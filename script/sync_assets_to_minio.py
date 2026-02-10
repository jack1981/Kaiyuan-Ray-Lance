#!/usr/bin/env python3

import argparse
import shutil
import tempfile
import zipfile
from pathlib import Path

import pyarrow.fs as pafs


def upload_file(fs: pafs.S3FileSystem, local_path: Path, remote_path: str) -> None:
    remote_parent = remote_path.rsplit("/", 1)[0]
    fs.create_dir(remote_parent, recursive=True)
    with local_path.open("rb") as src, fs.open_output_stream(remote_path) as dst:
        shutil.copyfileobj(src, dst, length=8 * 1024 * 1024)
    print(f"Uploaded file: {local_path} -> s3a://{remote_path}")


def upload_directory(fs: pafs.S3FileSystem, local_dir: Path, remote_prefix: str) -> None:
    if not local_dir.is_dir():
        raise FileNotFoundError(f"Missing directory: {local_dir}")

    files = sorted([p for p in local_dir.rglob("*") if p.is_file()])
    if not files:
        raise RuntimeError(f"No files found in directory: {local_dir}")

    for file_path in files:
        rel = file_path.relative_to(local_dir).as_posix()
        remote_path = f"{remote_prefix.rstrip('/')}/{rel}"
        upload_file(fs, file_path, remote_path)


def build_tiny_seq_classifier_archive(source_dir: Path) -> Path:
    if not source_dir.is_dir():
        raise FileNotFoundError(f"Missing model directory: {source_dir}")

    tmpdir = Path(tempfile.mkdtemp(prefix="tiny-seq-classifier-"))
    archive_path = tmpdir / "tiny_seq_classifier.zip"

    with zipfile.ZipFile(archive_path, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        for file_path in sorted(p for p in source_dir.rglob("*") if p.is_file()):
            rel = file_path.relative_to(source_dir.parent).as_posix()
            zf.write(file_path, arcname=rel)

    return archive_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Sync local example assets into MinIO for Ray-on-K8s"
    )
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--endpoint", default="host.docker.internal:30900")
    parser.add_argument("--scheme", default="http", choices=["http", "https"])
    parser.add_argument("--bucket", default="kaiyuan-ray")
    parser.add_argument("--access-key", default="minio")
    parser.add_argument("--secret-key", default="minio123")
    parser.add_argument("--region", default="us-east-1")
    args = parser.parse_args()

    data_root = Path(args.data_root)
    sample_root = data_root / "sample"
    models_root = data_root / "models"

    required_lance_dirs = [
        sample_root / "pcmind_kaiyuan_2b_sample.lance",
        sample_root / "scored_input.lance",
        sample_root / "fineweb_chinese.lance",
        sample_root / "dclm_subset.lance",
        sample_root / "dclm_subset_dedup.lance",
    ]
    required_model_files = [
        models_root / "fasttext_hq.bin",
        models_root / "fasttext_mmlu.bin",
    ]
    tiny_seq_dir = models_root / "tiny_seq_classifier"

    for path in required_lance_dirs:
        if not path.is_dir():
            raise FileNotFoundError(f"Missing Lance sample directory: {path}")

    for path in required_model_files:
        if not path.is_file():
            raise FileNotFoundError(f"Missing model file: {path}")

    fs = pafs.S3FileSystem(
        access_key=args.access_key,
        secret_key=args.secret_key,
        endpoint_override=args.endpoint,
        scheme=args.scheme,
        region=args.region,
        force_virtual_addressing=False,
    )

    bucket = args.bucket
    fs.create_dir(bucket, recursive=True)
    fs.create_dir(f"{bucket}/sample", recursive=True)
    fs.create_dir(f"{bucket}/output", recursive=True)
    fs.create_dir(f"{bucket}/models", recursive=True)
    fs.create_dir(f"{bucket}/ray-artifacts", recursive=True)

    for local_dir in required_lance_dirs:
        remote_prefix = f"{bucket}/sample/{local_dir.name}"
        upload_directory(fs, local_dir, remote_prefix)

    upload_file(fs, models_root / "fasttext_hq.bin", f"{bucket}/models/fasttext_hq.bin")
    upload_file(fs, models_root / "fasttext_mmlu.bin", f"{bucket}/models/fasttext_mmlu.bin")

    archive_path = build_tiny_seq_classifier_archive(tiny_seq_dir)
    try:
        upload_file(fs, archive_path, f"{bucket}/models/tiny_seq_classifier.zip")
    finally:
        shutil.rmtree(archive_path.parent, ignore_errors=True)

    print("MinIO sync completed.")


if __name__ == "__main__":
    main()
