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

    args = parser.parse_args()
    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    pipeline = PipelineTree(config, mode=args.mode, ray_address=args.ray_address)
    ds = pipeline.run()
    ds.show(20)
    print(f"row_count={ds.count()}")
