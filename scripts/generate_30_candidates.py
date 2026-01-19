#!/usr/bin/env python3
"""Generate candidates.json for all 30 images from BENCHMARK_ANALYSIS.md"""

import json

# All 30 images from BENCHMARK_ANALYSIS.md with their benchmark commands
# Format: (commit, pr, subject, benchmark_cmd, model, parent_commit_if_known)

TIER1_HIGH_IMPACT = [
    ("2a754e57", "#579", "2x prefill", "bench_latency --batch-size 1 --input-len 8192 --output-len 1", "meta-llama/Llama-3.1-8B-Instruct", None),
    ("9216b106", "#394", "40% scheduler", "bench_serving --num-prompt 300 --request-rate 16", "meta-llama/Llama-3.1-8B-Instruct", None),
    ("b1e5a33a", "#6960", "13% LoRA ITL", "bench_serving --num-prompt 480 --request-rate 8 --disable-radix-cache", "meta-llama/Llama-3.1-8B-Instruct", None),
    ("79961afa", "#6077", "21% FA3 faster", "bench_latency --batch-size 64 --input-len 2048 --output-len 256", "meta-llama/Llama-3.1-8B-Instruct", "cfca4e0e"),
    ("3212c2ad", "#6003", "16% VLM faster", "bench_serving --model llava-hf/llava-1.5-7b-hf --num-prompt 100 --request-rate 2", "llava-hf/llava-1.5-7b-hf", "53475674"),
    ("c087ddd6", "#6627", "10-15% kernel", "bench_latency --batch-size 32 --input-len 1024 --output-len 128", "meta-llama/Llama-3.1-8B-Instruct", "f4a8987f"),
    ("1acca3a2", "#5969", "FA3 len() removal", "bench_latency --batch-size 64 --input-len 2048 --output-len 256", "meta-llama/Llama-3.1-8B-Instruct", None),
]

PREVIOUSLY_REBUILT = [
    ("2bd18e2d", "#2901", "Memory pool", "bench_serving --num-prompt 200 --request-rate 8", "meta-llama/Llama-2-7b-hf", "83452dbb"),
    ("d1112d85", "#2797", "Input embeds", "bench_latency --batch-size 32 --input-len 512 --output-len 128", "google/gemma-2-2b", "48efec7b"),
    ("10189d08", "#2171", "CPU affinity", "bench_serving --num-prompt 200 --request-rate 16", "meta-llama/Llama-3.1-8B-Instruct", None),
    ("ddcf9fe3", "#3731", "Triton attention", "bench_serving --num-prompt 200 --request-rate 8", "meta-llama/Llama-2-7b-chat-hf", "6252ade9"),
    ("93470a14", "#5090", "FA3 optimization", "bench_latency --batch-size 64 --input-len 2048 --output-len 256", "meta-llama/Llama-3.1-8B-Instruct", "db452760"),
    ("f4a8987f", "-", "Parent baseline", "bench_serving --num-prompt 200 --request-rate 8", "meta-llama/Llama-3.1-8B-Instruct", None),
]

NEWLY_BUILT = [
    ("09deb20d", "#420", "Logits memory", "bench_serving --num-prompt 300 --request-rate 10", "meta-llama/Llama-3.1-8B-Instruct", None),
    ("2854a5ea", "#1496", "bench_latency fix", "bench_latency --batch-size 32 --input-len 512 --output-len 128", "meta-llama/Llama-3.1-8B-Instruct", None),
    ("564a898a", "#619", "Mem indices", "bench_serving --num-prompt 300 --request-rate 10", "meta-llama/Llama-3.1-8B-Instruct", None),
    ("62757db6", "#1010", "Cache disabled", "bench_serving --num-prompt 200 --request-rate 8 --disable-radix-cache", "meta-llama/Llama-3.1-8B-Instruct", None),
    ("6a2941f4", "#625", "TP overhead", "bench_serving --num-prompt 200 --request-rate 10", "meta-llama/Llama-3.1-8B-Instruct", None),
    ("6f560c76", "#117", "First token latency", "bench_serving --num-prompt 100 --request-rate 4 --output-len 256", "meta-llama/Llama-3.1-8B-Instruct", None),
    ("8f8f96a6", "#1773", "stop_token_ids", "bench_latency --batch-size 64 --input-len 512 --output-len 128", "meta-llama/Llama-3.1-8B-Instruct", None),
    ("9183c23e", "#2695", "Weights update", "bench_latency --batch-size 32 --input-len 512 --output-len 128", "meta-llama/Llama-3.1-8B-Instruct", None),
    ("9c064bf7", "#1587", "LoRA Step 1", "bench_serving --num-prompt 480 --request-rate 8 --disable-radix-cache", "meta-llama/Llama-3.1-8B-Instruct", None),
    ("9c745d07", "#2056", "xgrammar", "bench_serving --num-prompt 100 --request-rate 4", "meta-llama/Llama-3.1-8B-Instruct", None),
    ("ab4a83b2", "#1339", "Optimize schedule", "bench_serving --num-prompt 300 --request-rate 16", "meta-llama/Llama-3.1-8B-Instruct", None),
    ("ac971ff6", "#658", "stream_interval", "bench_serving --num-prompt 100 --request-rate 4 --output-len 512", "meta-llama/Llama-3.1-8B-Instruct", None),
    ("b1709305", "#1697", "Radix tree", "bench_serving --num-prompt 500 --request-rate 10", "meta-llama/Llama-3.1-8B-Instruct", None),
    ("b77a02cd", "#1752", "Grammar backends", "bench_serving --num-prompt 100 --request-rate 4", "meta-llama/Llama-3.1-8B-Instruct", None),
    ("c98e84c2", "#1589", "torch.argmax", "bench_latency --batch-size 64 --input-len 256 --output-len 256", "meta-llama/Llama-3.1-8B-Instruct", None),
    ("e3ec6bf4", "#6814", "FP8 quant", "bench_latency --batch-size 32 --input-len 1024 --output-len 128", "meta-llama/Llama-3.1-8B-Instruct", None),
    ("e5db40dc", "#1694", "ORJson", "bench_serving --num-prompt 500 --request-rate 30 --output-len 64", "meta-llama/Llama-3.1-8B-Instruct", None),
]

def generate_candidates():
    candidates = []
    all_images = TIER1_HIGH_IMPACT + PREVIOUSLY_REBUILT + NEWLY_BUILT

    for commit, pr, subject, bench_cmd, model, parent in all_images:
        # Build perf_command with python prefix
        if bench_cmd.startswith("bench_"):
            perf_command = f"python3 -m sglang.{bench_cmd}"
        else:
            perf_command = bench_cmd

        # Add model if not already in command
        if "--model" not in perf_command and "llava" not in perf_command:
            perf_command += f" --model {model}"

        candidate = {
            "human": commit,
            "parent": parent if parent else commit,  # Use self if no parent
            "human_full": commit,
            "parent_full": parent if parent else commit,
            "model": model,
            "perf_command": perf_command,
            "subject": f"{subject} ({pr})",
            "pr_url": f"https://github.com/sgl-project/sglang/pull/{pr.replace('#', '')}" if pr != "-" else "",
            "docker_repo": "shikhar481/sglang-images",
            "status": "ready",
            "lora_server_args": "--lora-paths lora=algoprog/fact-generation-llama-3.1-8b-instruct-lora" if "LoRA" in subject else "",
        }
        candidates.append(candidate)

    return candidates

if __name__ == "__main__":
    candidates = generate_candidates()
    with open("/tmp/sglang_3way_candidates.json", "w") as f:
        json.dump(candidates, f, indent=2)
    print(f"Generated {len(candidates)} candidates to /tmp/sglang_3way_candidates.json")

    # Print summary
    print("\nSummary by tier:")
    print(f"  Tier 1 (High-Impact): {len(TIER1_HIGH_IMPACT)} images")
    print(f"  Previously Rebuilt: {len(PREVIOUSLY_REBUILT)} images")
    print(f"  Newly Built: {len(NEWLY_BUILT)} images")
    print(f"  Total: {len(candidates)} images")
