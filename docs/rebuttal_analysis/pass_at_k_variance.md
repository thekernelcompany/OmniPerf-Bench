# W1 / kNyS scale — Rollout variance (pass@k campaign, generated)

Source: `Inferencebench/iso-bench-pass-at-k-results` (public). Independent agent rollouts per task, each rollout's patch benchmarked once on H100. Rollouts are fully isolated (`ISO-Bench/scripts/PASS_AT_K_PLAN.md`: worktree and state destroyed between samples), so spread across rollouts = agent stochasticity + benchmark noise combined.

## 1. Campaign coverage

| Repo | Agent | Tasks | Benchmarked rollouts | Rollouts/task (median) |
|---|---|---|---|---|
| sglang | claude_code | 10 | 76 | 8 |
| sglang | codex_cli | 10 | 74 | 8 |
| vllm | claude_code | 30 | 239 | 8 |
| vllm | codex_cli | 29 | 229 | 8 |

**618 benchmarked rollouts** across 40 distinct tasks (30 vLLM + 10 SGLang) and 2 agents. Appendix G (Table 9) currently reports only pass@1 vs pass@2 for these agents on 30 vLLM tasks — the collected campaign is considerably larger than what the paper shows.

## 2. Dispersion of the classifying metric across rollouts

For a fixed task the human reference does not change, so rollout-to-rollout variation in the agent metric *is* the variation in the agent-vs-human ratio that decides Beats/Similar/Worse (+/-5% band). `max dev` = largest deviation of any rollout from that task's rollout median. Metric per cell = the canonical `primary_metric` where the canonical table has one, else the family that repo's canonical classifications use (vLLM TTFT, SGLang throughput), else whatever the benchmark mode emits.

### By metric family

| Metric family | Cells | median CV | median max dev | p90 max dev | all rollouts within +/-5% |
|---|---|---|---|---|---|
| throughput | 34 | 0.5% | 1.0% | 4.7% | 31/34 |
| ttft | 45 | 1.0% | 1.9% | 19.2% | 34/45 |

### By repo and agent

| Repo | Agent | Cells | median max dev | p90 max dev | within +/-5% | within +/-2.5% |
|---|---|---|---|---|---|---|
| sglang | claude_code | 10 | 1.6% | 2.8% | 10/10 | 9/10 |
| sglang | codex_cli | 10 | 0.7% | 1.8% | 10/10 | 10/10 |
| vllm | claude_code | 30 | 2.4% | 34.8% | 20/30 | 17/30 |
| vllm | codex_cli | 29 | 1.4% | 12.3% | 25/29 | 20/29 |

Pooled over all 79 (task, agent) cells: median max deviation 1.7%, p90 14.1%; **65/79 (82%) have every rollout inside the +/-5% band**. The tail is concentrated in TTFT-classified tasks.

Least stable cells:

| Repo | Agent | Commit | Metric | n | max dev |
|---|---|---|---|---|---|
| vllm | claude_code | `19d98e0c` | ttft_mean_ms | 8 | 73.5% |
| vllm | claude_code | `9f1710f1` | ttft_mean_ms | 8 | 70.0% |
| vllm | codex_cli | `19d98e0c` | ttft_mean_ms | 8 | 67.8% |
| vllm | codex_cli | `89a84b0b` | ttft_mean_ms | 7 | 61.1% |
| vllm | claude_code | `98f47f2a` | throughput_tok_s | 8 | 34.8% |
| vllm | claude_code | `a3223766` | ttft_mean_ms | 8 | 19.2% |
| vllm | claude_code | `3b61cb45` | throughput_tok_s | 8 | 14.5% |
| vllm | claude_code | `8c1e77fb` | throughput_tok_s | 8 | 14.1% |

## 3. How many published classifications are robust to which rollout is scored

`boundary margin` = distance from the canonical `primary_pct` to the nearest +/-5% boundary. A cell is *at risk* when the observed rollout spread exceeds that margin. Cross-campaign, so read as a sensitivity estimate.

- Cells with a canonical percentage to compare against: **62**
- Robust: **50** (81%)
- At risk: **12** (19%)

| Repo | Agent | Commit | canonical % | class | margin | max dev |
|---|---|---|---|---|---|---|
| vllm | claude_code | `98f47f2a` | +5.3% | beats | 0.3% | 34.8% |
| vllm | claude_code | `b690e348` | -5.6% | worse | 0.6% | 1.9% |
| vllm | codex_cli | `e7b20426` | -5.9% | worse | 0.9% | 1.4% |
| vllm | claude_code | `9badee53` | -3.6% | similar | 1.4% | 5.4% |
| sglang | claude_code | `132dad87` | +3.4% | similar | 1.6% | 2.1% |
| vllm | claude_code | `58eee5f2` | -3.0% | similar | 2.0% | 2.5% |
| vllm | claude_code | `a3223766` | +8.2% | beats | 3.2% | 19.2% |
| vllm | claude_code | `3476ed08` | -1.6% | similar | 3.4% | 3.7% |
| vllm | claude_code | `299ebb62` | -1.5% | similar | 3.5% | 5.7% |
| vllm | claude_code | `9f1710f1` | +0.4% | similar | 4.6% | 70.0% |
| vllm | codex_cli | `19d98e0c` | +12.8% | beats | 7.8% | 67.8% |
| vllm | claude_code | `19d98e0c` | -59.4% | worse | 54.4% | 73.5% |

## 4. INTERNAL — do not put this in a reviewer response

17 (task, agent) cells carry no agent measurement in the canonical table (`primary_metric = agent_failed`, scored as Worse and therefore as a failed task in Tables 3-5). Of those, **14 used the identical benchmark command in the pass@k campaign and produced 7-8 successfully benchmarked patches there**:

| Repo | Agent | Commit | canonical | failure_reason | rollouts benchmarked |
|---|---|---|---|---|---|
| sglang | claude_code | `1acca3a2` | worse | failed | 8/8 |
| sglang | claude_code | `73b13e69` | worse | patch_failed | 8/8 |
| sglang | claude_code | `a191a0e4` | worse | patch_failed | 8/8 |
| sglang | claude_code | `da47621c` | worse | patch_failed | 8/8 |
| sglang | claude_code | `e3ec6bf4` | worse | patch_failed | 8/8 |
| vllm | claude_code | `3b61cb45` | worse | None | 8/8 |
| vllm | claude_code | `660470e5` | worse | None | 8/8 |
| vllm | claude_code | `8c1e77fb` | worse | None | 8/8 |
| vllm | claude_code | `9ed82e70` | worse | None | 8/8 |
| vllm | claude_code | `d7740ea4` | worse | None | 8/8 |
| vllm | codex_cli | `3476ed08` | worse | None | 8/8 |
| vllm | codex_cli | `9ed82e70` | worse | None | 8/8 |
| vllm | codex_cli | `a3223766` | worse | None | 8/8 |
| vllm | codex_cli | `d7740ea4` | worse | None | 7/8 |

Two readings, and the artifacts cannot separate them: (a) the single canonical rollout genuinely failed and failure is not reproducible, or (b) the canonical aggregate is missing a measurement that was never persisted — for the vLLM cells `failure_reason` is null and no `*_agent_result.json` exists under `benchmark_results/vllm/claude_code/`, while agent results for the same commits exist for other tools. Either way, these cells count against the agent in the published rates. Worth resolving internally before the data release; not worth volunteering to a reviewer.

## Caveats (do not drop when quoting these numbers)

- **The spread mixes two sources** — a different patch each rollout (agent stochasticity) and benchmark noise. No patch was benchmarked twice, so they cannot be separated; these are an upper bound on benchmark noise alone.
- **Absolute levels are not comparable to the canonical campaign.** The pass@k campaign reused each PR's benchmark command verbatim for 28 of 40 tasks; the other 12 have substitutions (unavailable model swapped, dtype/flag changes), and measured levels differ from `hard_metrics.json` regardless (e.g. `015069b0`: canonical human TTFT 13.7 ms vs ~600 ms here). Only *relative* dispersion transfers between campaigns.
- **Per-rollout Beats/Similar/Worse is not computable here**, so pass@k for k>2 cannot be reported from this dataset: it benchmarked agent patches only, with no human or baseline run in the same context. One human-patch benchmark per task (~40 runs, 1xH100) would unlock Hard Success pass@k up to k=8 on 40 tasks x 2 agents; True Success pass@k would additionally need a judge pass over the 618 rollout patches (API cost, no GPU).
- Appendix G's pass@2 table rests on its own separate 3-rollout evaluation on 30 vLLM tasks and is unaffected by anything here.

