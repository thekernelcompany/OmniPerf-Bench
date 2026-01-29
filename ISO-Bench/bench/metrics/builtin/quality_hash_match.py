import hashlib
from pathlib import Path
from typing import Dict, Any
from ..registry import register


def _sha256(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


@register("quality:hash_match")
def hash_match(spec: Dict[str, Any], candidate: str, cwd: str) -> Dict[str, Any]:
    args = spec.get("args", {})
    ref = Path(cwd) / args["ref"]
    cand = Path(cwd) / args["candidate"].format(candidate=candidate)
    
    ok = _sha256(ref) == _sha256(cand)
    return {
        "name": spec["name"], 
        "candidate": candidate, 
        "value": 1.0 if ok else 0.0, 
        "unit": "bool"
    }