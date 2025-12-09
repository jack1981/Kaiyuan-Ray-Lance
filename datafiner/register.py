import importlib

CLASS_REGISTRY = {}


def register(name, required_packages=None):
    def decorator(cls):
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
