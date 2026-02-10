import argparse
import os

import yaml

from datafiner.base import PipelineTree


def _rewrite_path_for_k8s(path: str, bucket: str) -> str:
    normalized = path.replace("\\", "/")
    if normalized.startswith(("s3://", "s3a://", "gs://", "abfs://")):
        return path

    candidate = normalized
    if candidate.startswith("lance.`") and candidate.endswith("`"):
        candidate = candidate[len("lance.`") : -1].replace("``", "`")
    elif candidate.startswith("lance."):
        candidate = candidate[len("lance.") :]

    mappings = (
        ("data/sample/", "sample/"),
        ("/workspace/data/sample/", "sample/"),
        ("workspace/data/sample/", "sample/"),
        ("data/output/", "output/"),
        ("/workspace/data/output/", "output/"),
        ("workspace/data/output/", "output/"),
        ("data/models/", "models/"),
        ("/workspace/data/models/", "models/"),
        ("workspace/data/models/", "models/"),
    )

    candidates = (
        candidate,
        candidate.lstrip("./"),
        candidate.lstrip("/"),
    )
    for current in candidates:
        for src_prefix, dest_prefix in mappings:
            if current.startswith(src_prefix):
                remainder = current[len(src_prefix) :].lstrip("/")
                if dest_prefix == "models/" and (
                    remainder == "tiny_seq_classifier"
                    or remainder.startswith("tiny_seq_classifier/")
                ):
                    return f"s3a://{bucket}/models/tiny_seq_classifier.zip"
                return f"s3a://{bucket}/{dest_prefix}{remainder}"
    return path


def _rewrite_config_paths_for_k8s(value, bucket: str):
    if isinstance(value, dict):
        return {k: _rewrite_config_paths_for_k8s(v, bucket) for k, v in value.items()}
    if isinstance(value, list):
        return [_rewrite_config_paths_for_k8s(v, bucket) for v in value]
    if isinstance(value, str):
        return _rewrite_path_for_k8s(value, bucket)
    return value


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run a Ray + Lance data pipeline.")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config.")
    parser.add_argument(
        "--mode",
        type=str,
        choices=["local", "k8s"],
        default="local",
        help="Execution mode. Use 'k8s' when connecting to a Ray cluster.",
    )
    parser.add_argument(
        "--ray-address",
        type=str,
        default=None,
        help="Optional Ray address, e.g. ray://<head-svc>:10001",
    )
    parser.add_argument(
        "--debug-stats",
        action="store_true",
        help="Enable Ray Data stage timing and ds.stats instrumentation logs.",
    )

    args = parser.parse_args()
    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    if args.mode == "k8s":
        bucket = os.getenv("K8S_DATA_BUCKET") or os.getenv("MINIO_BUCKET") or "kaiyuan-ray"
        config = _rewrite_config_paths_for_k8s(config, bucket)

    config.setdefault("ray", {})
    if args.debug_stats:
        config["ray"]["debug_stats"] = True

    pipeline = PipelineTree(config, mode=args.mode, ray_address=args.ray_address)
    ds = pipeline.run()
    ds.show(20)
    row_count = ds.count()
    print(f"row_count={row_count}")
    if config["ray"].get("debug_stats"):
        print(f"[RayDebug] final dataset stats\n{ds.stats()}")
