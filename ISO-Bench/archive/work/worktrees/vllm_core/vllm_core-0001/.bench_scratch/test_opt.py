import time
import random
from typing import Dict

# Benchmark DeepSeekR1ReasoningParser.extract_reasoning_content
from vllm.entrypoints.openai.reasoning_parsers.deepseek_r1_reasoning_parser import (
    DeepSeekR1ReasoningParser,
)


class DummyTokenizer:
    def __init__(self, vocab: Dict[str, int]):
        self._vocab = dict(vocab)

    def get_vocab(self) -> Dict[str, int]:
        return self._vocab


def make_samples():
    # Construct a variety of model outputs
    think_start = "<think>"
    think_end = "</think>"
    reason = "reasoning step " * 5
    answer = "final answer " * 3

    samples = [
        f"{think_start}{reason}{think_end}{answer}",  # normal case
        f"{reason}{think_end}{answer}",               # missing start token
        f"{think_start}{reason}",                     # no end token
        "no tags at all; plain output",               # no tags
    ]

    # Add some longer variants
    long_reason = ("R" * 512) + " middle " + ("R" * 512)
    long_answer = ("A" * 256) + " end " + ("A" * 256)
    samples += [
        f"{think_start}{long_reason}{think_end}{long_answer}",
        f"{long_reason}{think_end}{long_answer}",
    ]
    return samples


def bench_once(parser, samples, iters: int = 100_000):
    # Randomize inputs to avoid accidental caching effects
    rng = random.Random(123)
    order = [rng.randrange(len(samples)) for _ in range(iters)]
    start = time.perf_counter()
    res = None
    for idx in order:
        res = parser.extract_reasoning_content(samples[idx], request=None)
    end = time.perf_counter()
    # Return duration and a dummy use of res to prevent optimization
    return end - start, res


def main():
    tok = DummyTokenizer({"<think>": 1, "</think>": 2})
    parser = DeepSeekR1ReasoningParser(tok)
    samples = make_samples()

    # Warmup
    _ = bench_once(parser, samples, iters=10_000)

    # Timed run
    dur, res = bench_once(parser, samples, iters=100_000)
    print({"duration_sec": round(dur, 4), "last_result": res})


if __name__ == "__main__":
    main()

