import subprocess
import shlex
import statistics
from typing import Dict, Any
from ..registry import register


@register("runtime:throughput")
def throughput(spec: Dict[str, Any], candidate: str, cwd: str) -> Dict[str, Any]:
    cmd = spec.get("args", {}).get("cmd")
    if not cmd:
        raise ValueError("throughput metric requires args.cmd (string)")
    
    warmups = int(spec["args"].get("warmups", 3))
    trials = int(spec["args"].get("trials", 10))
    duration = float(spec["args"].get("duration_s", 3))

    values = []
    for i in range(warmups + trials):
        # TestPack should ensure cmd prints a single float (e.g., imgs/s)
        out = subprocess.check_output(shlex.split(cmd), cwd=cwd).decode().strip().split()
        val = float(out[-1])
        if i >= warmups:
            values.append(val)
    
    mean = statistics.mean(values)
    return {
        "name": spec["name"], 
        "candidate": candidate, 
        "value": mean, 
        "unit": "units/s", 
        "raw": values
    }