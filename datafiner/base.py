from __future__ import annotations

import copy
import os
from abc import ABC, abstractmethod

import ray

from datafiner.dataset_utils import (
    RuntimeConfig,
    RuntimeDataConfig,
    configure_data_context,
    default_storage_options,
    log_dataset_stats,
    timed_stage,
)
from datafiner.register import CLASS_REGISTRY


class PipelineNode(ABC):
    """Base pipeline node for Ray Data pipelines."""

    def __init__(self, runtime: RuntimeConfig, child_configs: list | None = None):
        super().__init__()
        self.runtime = runtime
        self.children = []
        if child_configs is None:
            return

        for child_config in child_configs:
            child_cfg = copy.deepcopy(child_config)
            child_type = child_cfg.pop("type")
            if child_type not in CLASS_REGISTRY:
                raise KeyError(f"Unknown pipeline node type: {child_type}")
            self.children.append(CLASS_REGISTRY[child_type](runtime, **child_cfg))

    @abstractmethod
    def run(self):
        """Run the pipeline step and return a Ray dataset."""
        raise NotImplementedError


class PipelineTree:
    """Pipeline tree entrypoint with Ray runtime bootstrap."""

    def __init__(
        self,
        config: dict,
        mode: str = "local",
        ray_address: str | None = None,
    ):
        if "pipeline" not in config:
            raise ValueError("Config must contain a top-level 'pipeline' key.")

        runtime_config = config.get("ray") or {}
        app_name = runtime_config.get("app_name", "kaiyuan-ray-pipeline")
        debug_stats = bool(runtime_config.get("debug_stats", False))
        data_config = RuntimeDataConfig.from_dict(runtime_config.get("data"))

        configured_address = runtime_config.get("address")
        if mode == "k8s":
            effective_address = (
                ray_address
                or configured_address
                or os.getenv("RAY_ADDRESS")
                or "ray://raycluster-kaiyuan-head-svc:10001"
            )
        else:
            effective_address = ray_address or configured_address

        init_kwargs = dict(runtime_config.get("init_kwargs", {}))
        init_kwargs.setdefault("ignore_reinit_error", True)
        init_kwargs.setdefault("log_to_driver", True)

        runtime_env = runtime_config.get("runtime_env")
        if runtime_env is not None:
            init_kwargs["runtime_env"] = runtime_env

        namespace = runtime_config.get("namespace")
        if namespace:
            init_kwargs["namespace"] = namespace

        if effective_address:
            ray.init(address=effective_address, **init_kwargs)
        else:
            ray.init(**init_kwargs)

        configure_data_context(data_config)

        self.runtime = RuntimeConfig(
            app_name=app_name,
            mode=mode,
            ray_address=effective_address,
            storage_options=default_storage_options(runtime_config.get("storage_options")),
            debug_stats=debug_stats,
            data=data_config,
        )

        pipeline_config = copy.deepcopy(config["pipeline"])
        class_type = pipeline_config.pop("type")
        if class_type not in CLASS_REGISTRY:
            raise KeyError(f"Unknown pipeline node type: {class_type}")
        self.root = CLASS_REGISTRY[class_type](self.runtime, **pipeline_config)

    def run(self):
        with timed_stage(self.runtime, "pipeline.root"):
            ds = self.root.run()
        log_dataset_stats(self.runtime, ds, "pipeline.result")
        return ds
