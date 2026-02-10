#!/usr/bin/env python3
from __future__ import annotations

import argparse
import time

import ray
import yaml

from datafiner.base import PipelineTree


def main() -> None:
    parser = argparse.ArgumentParser(description="Manual Ray Data benchmark runner.")
    parser.add_argument("--config", default="example/read_write.yaml")
    parser.add_argument("--mode", choices=["local", "k8s"], default="local")
    parser.add_argument("--ray-address", default=None)
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument(
        "--debug-stats",
        action="store_true",
        help="Enable detailed stage timings and ds.stats during benchmark.",
    )
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    config.setdefault("ray", {})
    if args.debug_stats:
        config["ray"]["debug_stats"] = True

    for i in range(args.repeat):
        start = time.perf_counter()
        pipeline = PipelineTree(config, mode=args.mode, ray_address=args.ray_address)
        ds = pipeline.run()
        row_count = ds.count()
        elapsed = time.perf_counter() - start
        print(f"[bench] run={i + 1}/{args.repeat} elapsed={elapsed:.3f}s row_count={row_count}")

        if config["ray"].get("debug_stats"):
            print(f"[bench] run={i + 1} ds.stats\n{ds.stats()}")

        ray.shutdown()


if __name__ == "__main__":
    main()
