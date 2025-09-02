import os
import json
import math
import torch
import numpy as np
from typing import Dict, Any, Tuple, List

# Import the actual optimized modules from the commit
# This commit modifies SamplingTensors.from_lists inside:
# vllm/model_executor/sampling_metadata.py
try:
    from vllm.model_executor.sampling_metadata import SamplingTensors
except Exception as e:
    raise ImportError(
        "Failed to import vllm.model_executor.sampling_metadata.SamplingTensors. "
        "Ensure you are running this script inside the repository with the commit applied "
        "and that 'vllm' is importable. Original error: {}".format(e)
    )


def setup() -> Dict[str, Any]:
    """Create a realistic workload to exercise the numpy padding optimization path."""
    torch.manual_seed(42)
    np.random.seed(42)

    # Workload design notes:
    # - We want do_penalties=True, so we must provide non-empty prompt_tokens/output_tokens.
    # - Use many sequences with variable lengths to maximize padding work.
    # - Use a moderate max length to keep memory reasonable but highlight padding overhead.
    # - Keep device='cpu' to focus on CPU-side construction cost where numpy optimization applies.
    #
    # Dimensions chosen to mimic real scenarios while staying resource-friendly.
    vocab_size = 50257
    num_rows = 8192            # number of sequences (rows)
    max_prompt_len = 192       # variable lengths up to this
    max_output_len = 256
    dtype = torch.float16
    device = torch.device("cpu")  # keep CPU to isolate CPU padding/build cost

    # Generate variable-length token lists
    # Use a skewed length distribution to ensure a wide range of paddings.
    prompt_lengths = np.random.randint(0, max_prompt_len + 1, size=num_rows)
    output_lengths = np.random.randint(0, max_output_len + 1, size=num_rows)

    prompt_tokens: List[List[int]] = []
    output_tokens: List[List[int]] = []
    for i in range(num_rows):
        pl = prompt_lengths[i]
        ol = output_lengths[i]
        if pl > 0:
            prompt_tokens.append(np.random.randint(0, vocab_size, size=pl, dtype=np.int64).tolist())
        else:
            prompt_tokens.append([])
        if ol > 0:
            output_tokens.append(np.random.randint(0, vocab_size, size=ol, dtype=np.int64).tolist())
        else:
            output_tokens.append([])

    # Create other lists expected by from_lists. Their exact semantics are not critical for
    # measuring the padding path, but we provide reasonable, consistent sizes.
    # We'll align these lengths with the number of rows to reflect typical usage.
    temperatures = (np.random.rand(num_rows).astype(np.float32) * 0.7 + 0.3).tolist()
    top_ps = (np.random.rand(num_rows).astype(np.float32) * 0.2 + 0.8).tolist()  # mostly close to 1
    top_ks = np.random.randint(1, 128, size=num_rows, dtype=np.int32).tolist()
    min_ps = (np.random.rand(num_rows).astype(np.float32) * 0.05).tolist()
    presence_penalties = (np.random.rand(num_rows).astype(np.float32) * 0.5).tolist()
    frequency_penalties = (np.random.rand(num_rows).astype(np.float32) * 0.5).tolist()
    repetition_penalties = (np.random.rand(num_rows).astype(np.float32) * 0.5 + 0.75).tolist()

    # sample_indices length is typically number of tokens that will be sampled; we use num_rows.
    sample_indices = list(range(num_rows))

    # sampling_seeds should be a 2D list: [batch_size, n_seeds]
    # We'll use 4 seeds per sequence.
    n_seeds = 4
    sampling_seeds = np.random.randint(
        np.iinfo(np.int64).min // 2,
        np.iinfo(np.int64).max // 2,
        size=(num_rows, n_seeds),
        dtype=np.int64
    ).tolist()

    extra_seeds_to_generate = 0  # consistent with typical call from from_sampling_metadata

    return {
        'temperatures': temperatures,
        'top_ps': top_ps,
        'top_ks': top_ks,
        'min_ps': min_ps,
        'presence_penalties': presence_penalties,
        'frequency_penalties': frequency_penalties,
        'repetition_penalties': repetition_penalties,
        'sampling_seeds': sampling_seeds,
        'sample_indices': sample_indices,
        'prompt_tokens': prompt_tokens,
        'output_tokens': output_tokens,
        'vocab_size': vocab_size,
        'extra_seeds_to_generate': extra_seeds_to_generate,
        'device': device,
        'dtype': dtype,
    }


def experiment(data: Dict[str, Any]) -> Any:
    """
    Execute the performance-critical code path being optimized.
    Specifically invokes SamplingTensors.from_lists which now uses numpy
    to create padded token matrices for prompt_tokens and output_tokens.
    """
    st = SamplingTensors.from_lists(
        temperatures=data['temperatures'],
        top_ps=data['top_ps'],
        top_ks=data['top_ks'],
        min_ps=data['min_ps'],
        presence_penalties=data['presence_penalties'],
        frequency_penalties=data['frequency_penalties'],
        repetition_penalties=data['repetition_penalties'],
        sampling_seeds=data['sampling_seeds'],
        sample_indices=data['sample_indices'],
        prompt_tokens=data['prompt_tokens'],
        output_tokens=data['output_tokens'],
        vocab_size=data['vocab_size'],
        extra_seeds_to_generate=data['extra_seeds_to_generate'],
        device=data['device'],
        dtype=data['dtype'],
    )
    return st


def _extract_result_summary(st: Any, vocab_size: int) -> Dict[str, Any]:
    """Extract a compact summary from SamplingTensors for equivalence checks."""
    result: Dict[str, Any] = {}

    # prompt_tokens and output_tokens are tensors (possibly empty) on device.
    # We store shapes and a few sampled rows/columns to validate correctness.
    for name in ['prompt_tokens', 'output_tokens']:
        t: torch.Tensor = getattr(st, name)
        # t may be empty tensor if do_penalties=False; here it's True, but guard anyway.
        shape = tuple(t.shape)
        dtype = str(t.dtype)
        device = str(t.device)
        result[f'{name}_shape'] = shape
        result[f'{name}_dtype'] = dtype
        result[f'{name}_device'] = device

        # Take up to 3 sample rows uniformly across rows
        samples = []
        if t.numel() > 0 and t.ndim == 2 and shape[0] > 0 and shape[1] > 0:
            t_cpu = t.detach().cpu()
            rows = shape[0]
            cols = shape[1]

            # pick indices roughly at 0, mid, last
            sample_row_ids = sorted(set([
                0,
                rows // 2,
                max(0, rows - 1),
            ]))
            for r in sample_row_ids:
                row_arr = t_cpu[r]
                # collect first 5 and last 5 entries for padding checks
                head = row_arr[:min(5, cols)].tolist()
                tail = row_arr[max(0, cols - min(5, cols)):].tolist()
                samples.append({
                    'row_idx': int(r),
                    'head': head,
                    'tail': tail,
                })
        result[f'{name}_samples'] = samples

    # Save vocab for checking padding sentinel if needed
    result['vocab_size'] = int(vocab_size)
    return result


def store_result(result: Any, filepath: str) -> None:
    """Store essential properties for equivalence checking."""
    # result is a SamplingTensors instance
    summary = _extract_result_summary(result, vocab_size=getattr(result, 'top_ks').numel() and None or None)
    # Note: We cannot access vocab_size from the object directly; pass via arg
    # So instead, re-extract by inferring from sample tails if needed.
    # To avoid ambiguity, we won't infer here. We'll save only shapes/values.
    # The vocab_size will be included by run_test when calling this function.
    # Hence, overload store_result to expect a tuple (result, vocab_size)
    raise RuntimeError("store_result(result, filepath) should be called with (SamplingTensors, vocab_size).")


def store_result_with_vocab(result: Any, vocab_size: int, filepath: str) -> None:
    summary = _extract_result_summary(result, vocab_size=vocab_size)
    torch.save(summary, filepath)


def load_result(filepath: str) -> Any:
    """Load reference result for equivalence checking."""
    data = torch.load(filepath)
    return data


def check_equivalence(current_result: Any, reference_result: Any) -> None:
    """Check that current result is equivalent to reference properties."""
    # current_result is SamplingTensors; reference_result is a saved summary dict
    vocab_size = reference_result.get('vocab_size', None)

    for name in ['prompt_tokens', 'output_tokens']:
        t: torch.Tensor = getattr(current_result, name)
        shape = tuple(t.shape)
        dtype = str(t.dtype)
        device = str(t.device)

        ref_shape = tuple(reference_result[f'{name}_shape'])
        ref_dtype = reference_result[f'{name}_dtype']
        ref_device = reference_result[f'{name}_device']

        assert shape == ref_shape, f"{name} shape mismatch: {shape} vs {ref_shape}"
        assert dtype == ref_dtype, f"{name} dtype mismatch: {dtype} vs {ref_dtype}"
        # Device might differ across environments; allow mismatch if both are cpu-like
        # but we still assert equality for consistency in same env runs.
        assert device == ref_device, f"{name} device mismatch: {device} vs {ref_device}"

        # Validate sampled head/tail values
        ref_samples = reference_result[f'{name}_samples']
        if t.numel() > 0 and t.ndim == 2 and len(ref_samples) > 0:
            t_cpu = t.detach().cpu()
            cols = shape[1]
            for sample in ref_samples:
                r = int(sample['row_idx'])
                cur_head = t_cpu[r][:min(5, cols)].tolist()
                cur_tail = t_cpu[r][max(0, cols - min(5, cols)):].tolist()

                assert cur_head == sample['head'], f"{name} head values differ at row {r}"
                assert cur_tail == sample['tail'], f"{name} tail values differ at row {r}"

        # Optional: If vocab_size is present, ensure that any padding positions
        # (where tail values equal sentinel in reference) still equal sentinel.
        if vocab_size is not None and t.numel() > 0 and t.ndim == 2:
            t_cpu = t.detach().cpu()
            samples = reference_result[f'{name}_samples']
            for s in samples:
                tail = s['tail']
                # positions equal to vocab_size in reference tail must still be so
                for idx, v in enumerate(tail):
                    if v == vocab_size:
                        cur_val = t_cpu[s['row_idx']][t_cpu.shape[1] - len(tail) + idx].item()
                        assert cur_val == vocab_size, f"{name} padding sentinel mismatch at sample row {s['row_idx']}"


def run_test(eqcheck: bool = False, reference: bool = False, prefix: str = '') -> float:
    """
    Run the performance test and return execution time in milliseconds.

    Args:
        eqcheck: Whether to perform equivalence checking
        reference: Whether to store result as reference
        prefix: Prefix for reference files

    Returns:
        Average execution time in milliseconds
    """
    data = setup()

    # Choose timing method based on the actual device used in experiment
    exp_device = data['device']
    use_cuda_timing = (exp_device.type == 'cuda')

    # Warmup iterations
    warmup = 3
    iters = 10

    if use_cuda_timing:
        torch.cuda.synchronize()
        with torch.no_grad():
            for _ in range(warmup):
                _ = experiment(data)
        num_iterations = iters
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        with torch.no_grad():
            start_event.record()
            for _ in range(num_iterations):
                result = experiment(data)
            end_event.record()
        torch.cuda.synchronize()
        elapsed_time_ms = start_event.elapsed_time(end_event)
        avg_ms = elapsed_time_ms / num_iterations
    else:
        import time
        for _ in range(warmup):
            _ = experiment(data)
        num_iterations = iters
        start = time.time()
        for _ in range(num_iterations):
            result = experiment(data)
        end = time.time()
        avg_ms = (end - start) * 1000.0 / num_iterations

    # Handle equivalence checking
    if eqcheck or reference:
        final_result = experiment(data)  # fresh run for correctness check
        ref_path = f'{prefix}_reference.pt'
        if reference:
            # Save with vocab_size for padding sentinel checks
            store_result_with_vocab(final_result, data['vocab_size'], ref_path)
        elif eqcheck:
            if not os.path.exists(ref_path):
                raise FileNotFoundError(f"Reference file not found: {ref_path}")
            reference_result = load_result(ref_path)
            check_equivalence(final_result, reference_result)

    return avg_ms

# The harness will be automatically appended here with argument parsing