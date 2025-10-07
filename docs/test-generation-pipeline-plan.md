### Test Generation and Execution Plan (v4 Generators)

**ARCHITECTURAL CORRECTION (2024):** The original plan proposed nested uv environments inside bench-env, but this approach was flawed due to Python version conflicts. The corrected approach uses Docker containers for complete per-commit isolation, leveraging the existing Docker infrastructure in the codebase.

#### Context
- `commit_to_dataset.py` orchestrates dataset assembly and invokes the test script resolution pipeline.
- `src/test_scripts/generate_test_generators.py` produces or locates per-commit test-case-generator scripts in `misc/experiments/generated_test_generators_v4/`.
- v4 generators expose a `run_test(eqcheck: bool, reference: bool, prefix: str)` function and expect a CLI harness matching the contract defined in `src/collect/generate/prompt.py`.

#### Current State and Observations
- Bench environment is active; vLLM installed via `uv`.
- Reproduced outcomes by importing and invoking `run_test` directly in the active env:
  - `2a052011_test_case_generator.py`: `TypeError: fused_moe() got an unexpected keyword argument use_fp8`.
  - `2bb0489c_test_case_generator.py`: `TypeError: SamplingTensors.from_lists() got an unexpected keyword argument sampling_seeds`.
  - `2deb029d_test_case_generator.py`: `AttributeError: PrefixCachingBlockAllocator object has no attribute get_computed_block_ids`.
  - `8d75fe48_test_case_generator.py`: `AttributeError: SimpleNamespace has no attribute input_scale` (Fp8LinearMethod.apply API mismatch).
  - `0f40557a_test_case_generator.py`: `ModuleNotFoundError: cacheflow.cache_ops` (external CUDA extension not installed).
  - `2f192835_test_case_generator.py`: OK.
  - `3a243095_test_case_generator.py`: OK.

#### Root Causes
- API/version mismatches: Generators are tied to specific upstream commits. A single global vLLM install cannot satisfy mixed, incompatible signatures across commits.
- External dependency: `cacheflow.cache_ops` is not present; requires cloning/building the cacheflow repo at the referenced commit with CUDA toolchain.
- Capability constraints: CUTLASS FP8 path (8d75fe48) requires specific GPU capability and CUDA versions.

#### Plan
1. Resolve and clone target repo per commit
   - Parse each extraction JSON in `misc/experiments/commit_extractions_with_apis/` to determine upstream repo and commit SHA.
   - In `assemble_canonical`, clone the upstream repo into a per-commit temp workdir and checkout both base and head SHAs for timing.

2. Create per-commit Docker containers (CORRECTED APPROACH)
   - For each commit, build a Docker image with the exact vLLM version at that commit.
   - Use the existing Docker infrastructure in the codebase (`src/harness/environment/docker_utils.py`).
   - Each container gets complete isolation with proper CUDA support and dependencies.
   - Build images on-demand and cache them for performance.

3. Compile GPU/custom ops when required
   - Ensure CUDA toolkit/driver compatibility. Fail fast with clear diagnostics if unsupported.
   - For cacheflow (0f40557a), clone the cacheflow repo at the commit and build `cache_ops` to satisfy `from cacheflow import cache_ops`.

4. Inject a CLI harness when missing
   - Before execution, generate a thin wrapper script that imports the generator module and wires `main()` to the documented CLI flags (`output_file`, `--eqcheck`, `--reference`, `--file_prefix`).
   - This keeps execution uniform for timing and equivalence I/O.

5. Capability checks and gating
   - Detect at runtime:
     - CUDA availability; GPU name and SM capability.
     - CUTLASS FP8 requirements for 8d75fe48 (SM89/SM90 and CUDA 12.x thresholds).
   - If a required capability is missing, mark deterministically (e.g., CAPABILITY_UNSUPPORTED) and skip execution instead of raising exceptions.

6. Execute tests and capture timings
   - Run the injected harness with `--reference` to store reference payloads where applicable; then with `--eqcheck` to validate.
   - Use the same CLI across base/head/main as required by `commit_to_dataset.py`. Parse timings via existing helpers.

7. Logging and artifacts
   - Persist per-commit logs (install/build logs, test stdout/stderr) and any reference files in the workdir for auditability.
   - On success, keep compact summaries; on failure, retain full logs.

8. Integrate into `commit_to_dataset.py`
   - Extend `assemble_canonical` to:
     - Resolve repo and clone at base/head commits into a temp workspace.
     - Create and use a per-commit venv.
     - Optionally execute `setup_commands`/`install_commands` from config.
     - Inject and run the harness for the selected generator, rather than ad-hoc subprocesses.
   - Ensure duration changes are computed from the harness execution results.

9. Cacheflow-specific path (0f40557a)
   - Clone cacheflow at the commit, build its CUDA extension, then run the generator in an isolated venv. Gate on CUDA presence.

#### Example Commands (per-test, isolated)
- vLLM at commit (example for 2a052011):
```bash
uv venv .venv_2a052011
source .venv_2a052011/bin/activate
uv pip install "git+https://github.com/vllm-project/vllm@2a052011"
python misc/experiments/generated_test_generators_v4/2a052011_test_case_generator.py /tmp/out.txt --reference --file_prefix gso
deactivate
```
- Cacheflow build (0f40557a) outline:
```bash
uv venv .venv_cacheflow_0f40557a
source .venv_cacheflow_0f40557a/bin/activate
# in cloned cacheflow repo at the target SHA
uv pip install -e .
python /path/to/0f40557a_test_case_generator.py /tmp/out.txt --reference --file_prefix gso
deactivate
```

#### Risks and Prerequisites
- Requires compatible CUDA toolkit/driver and sufficient GPU memory for FP8 fused paths.
- Building custom CUDA ops can be time-consuming; ensure toolchain availability (nvcc) and pinned versions.
- Disk space and runtime overhead for per-commit environments and builds.

#### Success Criteria
- Each generator runs against its matching upstream commit without API/signature errors.
- Equivalence checks pass where reference files exist; otherwise, reference files are generated deterministically.
- Capability-limited tests are reported as unsupported without raising runtime exceptions.
- Timings are captured and recorded into the dataset assembly flow.
