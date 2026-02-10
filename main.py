import argparse
import yaml

from datafiner.base import PipelineTree

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
