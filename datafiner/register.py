"""Node registration helpers used by pipeline config deserialization.

Pipeline modules register classes into `CLASS_REGISTRY` so `PipelineNode`
construction can resolve string `type` values from YAML configs.
This keeps pipeline reconstruction deterministic from configs, matching the
report's reproducibility goal for dataset construction.
"""

import importlib

CLASS_REGISTRY = {}


def register(name, required_packages=None):
    """Return a decorator that conditionally registers a node class by name.

    Args:
        name: Registry key used in YAML `type` fields.
        required_packages: Optional iterable of importable package names.

    Returns:
        A class decorator.

    Side effects:
        Imports optional dependencies and mutates global `CLASS_REGISTRY`.

    Assumptions:
        Registration names are unique; later registrations with the same name
        intentionally overwrite earlier entries.
    """
    def decorator(cls):
        """Register `cls` when dependency checks pass.

        Inputs/outputs:
            Takes a class and returns the same class object.

        Side effects:
            Prints a skip message when an optional dependency is missing and
            conditionally updates `CLASS_REGISTRY`.

        Assumptions:
            Missing optional dependencies should keep module import usable rather
            than failing hard.
        """
        if required_packages is not None:
            for pkg in required_packages:
                try:
                    importlib.import_module(pkg)
                except ImportError:
                    print(f"[register] {name} skipped: requires '{pkg}' not installed.")
                    return cls
        CLASS_REGISTRY[name] = cls
        return cls

    return decorator
