import time
import statistics
from vllm.sequence import SequenceStatus, Sequence, SequenceGroup
import random
import torch
from vllm.sequence import HiddenStates



def time_fn(fn, iters: int = 1_000_000):
    t0 = time.perf_counter()
    fn()
    t1 = time.perf_counter()
    return t1 - t0


def bench_is_finished(iters=2_000_000):
    statuses = [
        SequenceStatus.WAITING,
        SequenceStatus.RUNNING,
        SequenceStatus.SWAPPED,
        SequenceStatus.FINISHED_STOPPED,
        SequenceStatus.FINISHED_LENGTH_CAPPED,
        SequenceStatus.FINISHED_ABORTED,
        SequenceStatus.FINISHED_IGNORED,
    ]
    idx = 0

    def _run():
        nonlocal idx
        s = statuses[idx % len(statuses)]
        for _ in range(iters):
            SequenceStatus.is_finished(s)
            idx += 1

    return time_fn(_run)


def bench_get_finished_reason(iters=2_000_000):
    statuses = [
        SequenceStatus.WAITING,
        SequenceStatus.RUNNING,
        SequenceStatus.SWAPPED,
        SequenceStatus.FINISHED_STOPPED,
        SequenceStatus.FINISHED_LENGTH_CAPPED,
        SequenceStatus.FINISHED_ABORTED,
        SequenceStatus.FINISHED_IGNORED,
    ]
    idx = 0

    def _run():
        nonlocal idx
        s = statuses[idx % len(statuses)]
        for _ in range(iters):
            SequenceStatus.get_finished_reason(s)
            idx += 1

    return time_fn(_run)


def bench_sequence_props(iters=500_000):
    # Minimal inputs for Sequence
    inputs = {
        "prompt": "hello",
        "prompt_token_ids": list(range(1000)),
    }
    seq = Sequence(seq_id=1, inputs=inputs, block_size=16)

    def _run():
        for _ in range(iters):
            _ = seq.n_blocks
            _ = seq.get_len()
            _ = seq.is_prefill()
    return time_fn(_run)


def bench_group_methods(iters=300_000):
    inputs = {
        "prompt": "hello",
        "prompt_token_ids": list(range(64)),
    }
    seqs = [Sequence(i, inputs, block_size=16) for i in range(8)]
    sg = SequenceGroup("r1", seqs, arrival_time=0.0)

    def _run():
        for _ in range(iters):
            _ = sg.is_finished()
            _ = sg.is_prefill()
            _ = sg.num_unfinished_seqs()
    return time_fn(_run)




def build_hidden_states(n=4000, d=32):
    hs = HiddenStates.__new__(HiddenStates)
    hs.seq_ids = list(range(n))
    hs.hidden_states = torch.randn(n, d)
    return hs


def bench_hidden_states_prune(runs=40, n=4000):
    def _run():
        hs = build_hidden_states(n, 32)
        new_ids = [i for i in range(n) if (i % 7) != 0]
        class DummySG:
            def __init__(self, ids):
                self.seq_data = ids
        sg_list = [DummySG(new_ids)]
        hs.prune(sg_list)
    return time_fn(_run, iters=runs)

if __name__ == "__main__":
    results = {}
    results["is_finished"] = bench_is_finished()
    results["get_finished_reason"] = bench_get_finished_reason()
    results["sequence_props"] = bench_sequence_props()
    results["group_methods"] = bench_group_methods()
    results["hidden_states_prune"] = bench_hidden_states_prune()

    print("Timings (s):")
    for k, v in results.items():
        print(f"{k}: {v:.6f}")
