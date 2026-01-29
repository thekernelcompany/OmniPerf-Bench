import time
import numpy as np

from vllm.v1.spec_decode.ngram_proposer import NgramProposer


def run_once(context_len=200_000, vocab=10000, n=6, k=32):
    # Create a random context
    rng = np.random.default_rng(123)
    context = rng.integers(0, vocab, size=context_len, dtype=np.int32)

    # Choose an earlier position that has at least k tokens following it
    insert_pos = context_len // 3
    if insert_pos + n + k >= context_len - n:
        insert_pos = max(0, (context_len - n - k - 1) // 2)

    # Make the last n tokens equal to the pattern starting at insert_pos,
    # guaranteeing a match earlier in the context.
    context[-n:] = context[insert_pos:insert_pos + n]

    proposer = NgramProposer()

    # Warmup
    _ = proposer.propose(context, n=n, k=k)

    iters = 50
    t0 = time.time()
    for _ in range(iters):
        _ = proposer.propose(context, n=n, k=k)
    t1 = time.time()
    return (t1 - t0) / iters


def main():
    # Run a couple of trials to stabilize numbers
    trials = 3
    times = []
    for _ in range(trials):
        times.append(run_once())
    print(f"Avg propose time over {trials} trials: {np.mean(times):.6f}s (std {np.std(times):.6f}s)")


if __name__ == "__main__":
    main()
