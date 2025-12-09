from abc import ABC, abstractmethod

from pyspark.conf import SparkConf
from pyspark.sql import SparkSession
from datafiner.register import CLASS_REGISTRY


class PipelineNode(ABC):
    """
    Base pipeline node.
    """

    def __init__(
        self,
        spark: SparkSession,
        child_configs: list = None,
    ):
        super().__init__()
        self.spark = spark
        self.children = []
        if child_configs is not None:
            for child_config in child_configs:
                child_type = child_config.pop("type")
                self.children.append(CLASS_REGISTRY[child_type](spark, **child_config))

    @abstractmethod
    def run(self):
        """
        Run the pipeline step.
        """
        pass


class PipelineTree:
    """
    Pipeline tree class.
    """

    def __init__(self, config: dict):
        spark_config = config["spark"]
        conf = SparkConf()
        conf.set("spark.app.name", spark_config["app_name"])
        self.spark = SparkSession.builder.config(conf=conf).getOrCreate()

        pipeline_config = config["pipeline"]
        class_type = pipeline_config.pop("type")
        self.root = CLASS_REGISTRY[class_type](self.spark, **pipeline_config)

    def run(self):
        return self.root.run()
