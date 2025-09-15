from __future__ import annotations
from typing import Callable, Dict, Any


_REGISTRY: Dict[str, Callable[..., dict]] = {}


def register(name: str):
    def deco(fn):
        if name in _REGISTRY:
            raise ValueError(f"Metric already registered: {name}")
        _REGISTRY[name] = fn
        return fn
    return deco


def run_metric(using: str, spec: Dict[str, Any], candidate: str, cwd: str) -> Dict[str, Any]:
    if using not in _REGISTRY:
        raise KeyError(f"Metric '{using}' not registered")
    return _REGISTRY[using](spec, candidate, cwd)