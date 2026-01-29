from importlib import import_module
from pathlib import Path
from types import ModuleType
from .contract import TestPack


def load_testpack(entrypoint: str) -> TestPack:
    if ":" not in entrypoint:
        # local path that exposes pack() in __init__.py or pack.py
        p = Path(entrypoint).resolve()
        if not p.exists():
            raise FileNotFoundError(f"TestPack path not found: {p}")
        
        import sys
        sys.path.insert(0, str(p))
        
        try:
            if (p / "pack.py").exists():
                mod = import_module("pack")
            else:
                mod = import_module("__init__")
            return getattr(mod, "pack")()
        finally:
            sys.path.remove(str(p))
    
    # Python entrypoint format: "package.module:function"
    mod_name, attr = entrypoint.split(":", 1)
    mod: ModuleType = import_module(mod_name)
    return getattr(mod, attr)()