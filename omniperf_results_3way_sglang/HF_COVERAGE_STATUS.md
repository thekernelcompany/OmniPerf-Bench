# HuggingFace SGLang Coverage Status

## Target Dataset
- **Dataset**: `Ayushnangia/omniperf_v1` (sglang config)
- **Total commits**: 67

## Current Status

### Previously Verified Working (9 commits)
These were part of the v04x-triton batch that was benchmarked successfully:
- 132dad874d2e, 1acca3a2c685, 6b231325b978, 6cb00c639812
- 79961afa8281, a191a0e47c2f, b1e5a33ae337, dd1012fcbe2a, df7f61ee7d23

### Building Now (58 commits)
Building v04x-triton images with fixed dependencies:
- triton==3.0.0
- compressed_tensors<0.13.0
- numpy<2.0

Check progress: `tail -f /tmp/hf_builds.log`
Check status: `cat /tmp/hf_build_status.csv`

### After Builds Complete
Run benchmarks: `/tmp/bench_all_hf.sh`

## Dockerfile Fix Applied
```dockerfile
FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04
# ... base setup ...
RUN pip install "triton==3.0.0"
# ... install sglang ...
RUN pip install "compressed-tensors<0.13.0" --force-reinstall
RUN pip install "numpy<2.0" --force-reinstall
RUN pip install "triton==3.0.0" --force-reinstall
```
