"""Core pipeline node abstractions and Ray runtime bootstrap utilities.

This module owns the pipeline tree construction flow and Ray initialization used
by all reader/transform/writer nodes.
The tree-structured, YAML-driven composition mirrors the reproducibility
requirements described in Section 3.2 of the PCMind-2.1 technical report.
See also `datafiner/register.py` for node type registration.
"""

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
    """Base class for all pipeline nodes participating in a tree execution.

    Inputs/outputs:
        Accepts runtime config and optional child-node configs at construction.
        Subclasses implement `run()` and return a Ray `Dataset`.

    Side effects:
        Instantiates child nodes recursively from config dictionaries.

    Assumptions:
        Every child config has a valid `type` key that exists in CLASS_REGISTRY.
    """

    def __init__(self, runtime: RuntimeConfig, child_configs: list | None = None):
        """Construct a node and recursively instantiate configured children.

        Args:
            runtime: Shared runtime settings for Ray execution/storage behavior.
            child_configs: Optional list of child node config dictionaries.

        Returns:
            None.

        Side effects:
            Mutates `self.children` by creating child node instances.

        Assumptions:
            Child config dictionaries contain a `type` key matching a registered
            node class.
        """
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
    """Pipeline entrypoint that initializes Ray and executes the root node.

    Inputs/outputs:
        Consumes YAML-derived config, chooses Ray connection mode, and exposes
        `run()` returning the final Ray `Dataset`.

    Side effects:
        Calls `ray.init(...)`, mutates Ray Data global context, and may print
        debug runtime metadata.

    Assumptions:
        Config contains a top-level `pipeline` section with a registered `type`.
    """

    def __init__(
        self,
        config: dict,
        mode: str = "local",
        ray_address: str | None = None,
    ):
        """Initialize Ray runtime settings and build the root pipeline node.

        Args:
            config: Full pipeline configuration dictionary.
            mode: Execution mode (`local` or `k8s`) controlling address defaults.
            ray_address: Optional explicit Ray address override.

        Returns:
            None.

        Side effects:
            Initializes Ray, configures Ray Data context, and instantiates the
            full node tree.

        Assumptions:
            Runtime options under `config["ray"]` are compatible with `ray.init`.
        """
        if "pipeline" not in config:
            raise ValueError("Config must contain a top-level 'pipeline' key.")

        runtime_config = config.get("ray") or {}
        app_name = runtime_config.get("app_name", "kaiyuan-ray-pipeline")
        debug_stats = bool(runtime_config.get("debug_stats", False))
        data_config = RuntimeDataConfig.from_dict(runtime_config.get("data"))

        configured_address = runtime_config.get("address")
        # NOTE(readability): K8s mode defaults to `auto` so in-cluster RayJobs
        # can discover the head service without hardcoding addresses.
        if mode == "k8s":
            effective_address = (
                ray_address
                or configured_address
                or os.getenv("RAY_ADDRESS")
                or "auto"
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
        if self.runtime.debug_stats:
            print(
                "[RayDebug] runtime "
                f"mode={self.runtime.mode} "
                f"ray_address={self.runtime.ray_address} "
                f"storage_options={self.runtime.storage_options}"
            )

        pipeline_config = copy.deepcopy(config["pipeline"])
        class_type = pipeline_config.pop("type")
        if class_type not in CLASS_REGISTRY:
            raise KeyError(f"Unknown pipeline node type: {class_type}")
        self.root = CLASS_REGISTRY[class_type](self.runtime, **pipeline_config)

    def run(self):
        """Execute the configured pipeline tree and return the resulting dataset.

        Inputs/outputs:
            No inputs; returns the final Ray `Dataset` produced by the root node.

        Side effects:
            Triggers distributed Ray tasks and optional debug stats logging.

        Assumptions:
            Child nodes maintain schema/runtime contracts expected by downstream
            nodes.
        """
        with timed_stage(self.runtime, "pipeline.root"):
            ds = self.root.run()
        log_dataset_stats(self.runtime, ds, "pipeline.result")
        return ds
