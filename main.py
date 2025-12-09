import argparse
import yaml

from datafiner.base import PipelineTree

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=None)

    args = parser.parse_args()
    config = yaml.load(open(args.config, "r"), Loader=yaml.FullLoader)

    pipeline = PipelineTree(config)
    df = pipeline.run()
    df.show()
    print(df.count())
