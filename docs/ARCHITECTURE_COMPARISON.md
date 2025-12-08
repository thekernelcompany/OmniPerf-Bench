# OmniPerf-Bench Architecture: A Complete Guide

> **Who is this for?** Anyone wanting to understand how this benchmark works, how GSO relates to it, and how to replicate or extend it. Written to be accessible to newcomers.

---

## Table of Contents

1. [The Big Picture: What Problem Are We Solving?](#1-the-big-picture-what-problem-are-we-solving)
2. [The Three Systems Explained](#2-the-three-systems-explained)
3. [How Everything Connects](#3-how-everything-connects)
4. [Data Formats: Speaking the Same Language](#4-data-formats-speaking-the-same-language)
5. [Step-by-Step Workflows](#5-step-by-step-workflows)
6. [Problems and Gaps](#6-problems-and-gaps)
7. [How to Replicate GSO](#7-how-to-replicate-gso)
8. [Quick Reference](#8-quick-reference)

---

## 1. The Big Picture: What Problem Are We Solving?

### The Simple Explanation

Imagine you're a teacher grading students on "how well can you make code run faster?"

**The challenge:**
1. You need **test questions** (optimization tasks with slow code)
2. You need **answer keys** (how expert developers actually fixed it)
3. You need a **grading system** (did the student's fix actually make it faster?)

**OmniPerf-Bench is the entire exam system:**
- **GSO** = The textbook + answer key creator
- **perf-agents-bench** = The exam room where AI students take the test
- **Harness** = The grading machine

### Why This Matters

We want to know: **"Can AI agents optimize real-world code?"**

To answer this, we need:
```
Real optimization problems → AI tries to solve them → We measure if it worked
```

---

## 2. The Three Systems Explained

### System 1: GSO (Global Software Optimization)

**What it is:** A complete framework for creating optimization benchmarks.

**Analogy:** GSO is like a **textbook publisher** that:
1. Finds interesting problems from real codebases (like a researcher finding good exam questions)
2. Creates tests to measure performance (like creating answer rubrics)
3. Packages everything into a standardized format (like publishing the textbook)

**What GSO does:**

```
┌─────────────────────────────────────────────────────────────────────┐
│                         GSO PIPELINE                                │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│   STEP 1: Find Performance Commits                                  │
│   ────────────────────────────────                                  │
│   "Which commits in this repo made things faster?"                  │
│                                                                     │
│   Git History ──► LLM Analysis ──► Performance Commits              │
│   (thousands)     (filtering)       (dozens)                        │
│                                                                     │
│   STEP 2: Map to APIs                                               │
│   ──────────────────                                                │
│   "What user-facing feature did this optimization affect?"          │
│                                                                     │
│   Commit ──► Code Analysis ──► "This speeds up torch.compile()"     │
│                                                                     │
│   STEP 3: Generate Tests                                            │
│   ─────────────────────                                             │
│   "How do we measure if someone recreates this speedup?"            │
│                                                                     │
│   API + Commit ──► LLM ──► Performance Test Script                  │
│                                                                     │
│   STEP 4: Execute & Measure                                         │
│   ───────────────────────                                           │
│   "How much faster did it actually get?"                            │
│                                                                     │
│   Run test on: BEFORE commit → 2.5 seconds                          │
│                AFTER commit  → 1.0 seconds                          │
│                             = 2.5x speedup!                         │
│                                                                     │
│   STEP 5: Build Dataset                                             │
│   ─────────────────────                                             │
│   Package into JSONL with all the info needed to:                   │
│   - Reproduce the environment                                       │
│   - Run the test                                                    │
│   - Grade submissions                                               │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

**GSO Output:** A `.jsonl` file with 100+ optimization tasks, each containing:
- The codebase state before optimization
- A test to measure performance
- The expert's solution (for comparison)
- Metadata about what speedup to expect

---

### System 2: perf-agents-bench

**What it is:** A framework for running AI agents on optimization tasks.

**Analogy:** perf-agents-bench is like an **exam proctoring system** that:
1. Sets up isolated exam rooms (git worktrees)
2. Gives students the exam (optimization prompts)
3. Watches them work (agent execution)
4. Collects their answers (patches)

**What perf-agents-bench does:**

```
┌─────────────────────────────────────────────────────────────────────┐
│                    PERF-AGENTS-BENCH PIPELINE                       │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│   PHASE 1: PLAN                                                     │
│   ────────────────                                                  │
│   "What commits should we test agents on?"                          │
│                                                                     │
│   Task Config ──► Resolve Commits ──► plan.json                     │
│   (vllm.yaml)     (find parents)      (list of items)               │
│                                                                     │
│   PHASE 2: PREPARE                                                  │
│   ─────────────────                                                 │
│   "Run each agent on each task"                                     │
│                                                                     │
│   For each task:                                                    │
│   ┌─────────────────────────────────────────────────────────────┐   │
│   │ 1. Create isolated workspace (git worktree)                 │   │
│   │    └── Agent can't break the main repo                      │   │
│   │                                                             │   │
│   │ 2. Generate prompt (task.txt)                               │   │
│   │    └── "Here's slow code, make it faster"                   │   │
│   │                                                             │   │
│   │ 3. Run agent (OpenHands / TRAE / Codex)                     │   │
│   │    └── Agent reads code, thinks, makes changes              │   │
│   │                                                             │   │
│   │ 4. Collect results                                          │   │
│   │    └── Extract the patch (diff of changes)                  │   │
│   │    └── Write journal.json (metadata)                        │   │
│   │    └── Write prediction.jsonl (GSO format)                  │   │
│   └─────────────────────────────────────────────────────────────┘   │
│                                                                     │
│   PHASE 3: REPORT                                                   │
│   ────────────────                                                  │
│   "Summarize what happened"                                         │
│                                                                     │
│   All Journals ──► Aggregate ──► Summary Report                     │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

**perf-agents-bench Output:**
- `journal.json` files (what the agent did)
- `model_patch.diff` files (the agent's code changes)
- `prediction.jsonl` files (GSO-compatible format for grading)

---

### System 3: Parent src/ (The Glue)

**What it is:** A thin wrapper that connects everything together.

**Analogy:** The parent `src/` is like the **school administrator** that:
- Uses GSO's textbooks
- Runs exams through perf-agents-bench
- Sends answers to the grading machine

**What's in parent src/:**

| File | Purpose |
|------|---------|
| `src/harness/opt_at_k.py` | Calls GSO's grading system |
| `src/collect/execute/skymgr.py` | Cloud execution (shared with GSO) |
| `src/test_scripts/generate_test_generators.py` | LLM test generation |

**Key insight:** Parent src/ is mostly a **thin wrapper** - the real work happens in GSO and perf-agents-bench.

---

## 3. How Everything Connects

### The Complete Flow

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                           COMPLETE SYSTEM FLOW                               │
├──────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌─────────────┐                                                             │
│  │ Git Repos   │                                                             │
│  │ (vLLM, etc) │                                                             │
│  └──────┬──────┘                                                             │
│         │                                                                    │
│         ▼                                                                    │
│  ╔═══════════════════════════════════════════════════════════╗               │
│  ║              DATASET CREATION (GSO)                       ║               │
│  ╠═══════════════════════════════════════════════════════════╣               │
│  ║  Extract Commits → Map APIs → Generate Tests → Measure    ║               │
│  ╚═══════════════════════════════════════════════════════════╝               │
│         │                                                                    │
│         ▼                                                                    │
│  ┌─────────────────────────────────────────────────────────────┐             │
│  │                    DATASET.jsonl                            │             │
│  │  • 100+ optimization tasks                                  │             │
│  │  • Performance tests                                        │             │
│  │  • Expert solutions (ground truth)                          │             │
│  │  • Environment specs                                        │             │
│  └────────────────────────┬────────────────────────────────────┘             │
│                           │                                                  │
│         ┌─────────────────┼─────────────────┐                                │
│         ▼                 ▼                 ▼                                │
│  ╔═════════════╗   ╔═════════════╗   ╔═════════════╗                         │
│  ║  OpenHands  ║   ║    TRAE     ║   ║   Codex    ║                          │
│  ║   Agent     ║   ║   Agent     ║   ║   Agent    ║                          │
│  ╚══════╤══════╝   ╚══════╤══════╝   ╚══════╤══════╝                         │
│         │                 │                 │                                │
│         │    ┌────────────┴────────────┐    │                                │
│         │    │   perf-agents-bench     │    │                                │
│         │    │   (agent orchestration) │    │                                │
│         │    └────────────┬────────────┘    │                                │
│         │                 │                 │                                │
│         └─────────────────┼─────────────────┘                                │
│                           ▼                                                  │
│  ┌─────────────────────────────────────────────────────────────┐             │
│  │                  predictions.jsonl                          │             │
│  │  • Agent's patches                                          │             │
│  │  • One per task per agent                                   │             │
│  └────────────────────────┬────────────────────────────────────┘             │
│                           │                                                  │
│                           ▼                                                  │
│  ╔═══════════════════════════════════════════════════════════╗               │
│  ║              EVALUATION (GSO Harness)                     ║               │
│  ╠═══════════════════════════════════════════════════════════╣               │
│  ║  Build Docker → Apply Patch → Run Tests → Compute Speedup ║               │
│  ╚═══════════════════════════════════════════════════════════╝               │
│                           │                                                  │
│                           ▼                                                  │
│  ┌─────────────────────────────────────────────────────────────┐             │
│  │                     RESULTS                                 │             │
│  │  • Opt@K metrics (success rate at K attempts)               │             │
│  │  • Speedup comparisons                                      │             │
│  │  • Per-agent breakdowns                                     │             │
│  └─────────────────────────────────────────────────────────────┘             │
│                                                                              │
└──────────────────────────────────────────────────────────────────────────────┘
```

### What Connects to What

```
                    ┌─────────────────────────────────────────┐
                    │           DATA FORMAT                    │
                    │                                          │
                    │  GSO-compatible JSONL is the             │
                    │  "universal language" that lets          │
                    │  all systems talk to each other          │
                    │                                          │
                    │  Fields:                                 │
                    │  • instance_id                           │
                    │  • model_patch                           │
                    │  • model_name_or_path                    │
                    │                                          │
                    └─────────────────────────────────────────┘
                                      │
           ┌──────────────────────────┼──────────────────────────┐
           │                          │                          │
           ▼                          ▼                          ▼
    ┌──────────────┐          ┌──────────────┐          ┌──────────────┐
    │     GSO      │          │ perf-agents  │          │   Parent     │
    │              │◄─────────│    -bench    │─────────►│    src/      │
    │  Creates     │ uses     │              │  uses    │              │
    │  datasets    │ format   │  Runs agents │  format  │  Wraps both  │
    └──────────────┘          └──────────────┘          └──────────────┘
```

---

## 4. Data Formats: Speaking the Same Language

### The Core Formats

#### 1. Dataset Record (GSO creates this)

```json
{
  "instance_id": "vllm__vllm-abc1234",
  "repo": "vllm-project/vllm",
  "base_commit": "parent_hash",
  "opt_commit": "optimization_hash",
  "efficiency_test": "def test_performance():\n    ...",
  "setup_commands": ["apt-get install ..."],
  "install_commands": ["pip install -e ."],
  "hints_text": "Optimize attention kernel",
  "gt_diff": "diff --git a/...",
  "created_at": "2025-01-15"
}
```

**Think of it as:** The exam question with answer key attached.

#### 2. Prediction Record (Agents produce this)

```json
{
  "instance_id": "vllm__vllm-abc1234",
  "model_patch": "diff --git a/file.py\n-old\n+new",
  "model_name_or_path": "claude-3-opus"
}
```

**Think of it as:** The student's answer sheet.

#### 3. Journal Record (perf-agents-bench produces this)

```json
{
  "task_id": "vllm_core",
  "item_id": "vllm_core-0001",
  "status": "success",
  "commits": {
    "pre": "before_hash",
    "human": "expert_hash"
  },
  "duration_s": 847.5,
  "metrics": {
    "commit_count": 3,
    "patch_size_loc": 45
  }
}
```

**Think of it as:** The exam proctor's notes about how the student did.

### How Formats Convert

```
perf-agents-bench                        GSO Harness
      │                                       │
      │  journal.json                         │
      │  + model_patch.diff                   │
      │         │                             │
      │         ▼                             │
      │  ┌─────────────────┐                  │
      │  │ prediction.jsonl │ ◄───────────────┤ Reads this
      │  │ (GSO format)     │                 │
      │  └─────────────────┘                  │
      │                                       │
      ▼                                       ▼
  Agent work done                    Grading begins
```

---

## 5. Step-by-Step Workflows

### Workflow A: Creating a Dataset (Like GSO Does)

**Goal:** Turn a git repository into a benchmark dataset.

```bash
# Step 1: Extract performance commits
# "Find commits that made things faster"
python src/gso/collect/analysis/commits.py config.yaml
# Output: List of commits with performance improvements

# Step 2: Map APIs
# "What user-facing features did these commits affect?"
python src/gso/collect/analysis/apis.py repo_name
# Output: Commit → API mapping

# Step 3: Generate tests
# "Create tests that measure the speedup"
python src/gso/collect/generate/generate.py config.yaml
# Output: Test scripts for each commit

# Step 4: Execute on cloud
# "Actually run the tests and measure timing"
python src/gso/collect/execute/execute.py --exp_id repo --machines 8
# Output: Timing data (before: 2.5s, after: 1.0s)

# Step 5: Build dataset
# "Package everything into final format"
python src/gso/collect/build_dataset.py --exp_id repo
# Output: dataset.jsonl with 100+ tasks
```

**Visual:**

```
Day 1: Extract         Day 2: Generate       Day 3: Execute        Day 4: Package
─────────────────      ──────────────────    ──────────────────    ──────────────────
     │                       │                     │                     │
     ▼                       ▼                     ▼                     ▼
┌─────────┐            ┌─────────┐           ┌─────────┐           ┌─────────┐
│ 10,000  │            │   50    │           │   50    │           │  100+   │
│ commits │──filter───►│ perf    │──tests───►│ tasks   │──score───►│ tasks   │
│         │            │ commits │           │ running │           │ dataset │
└─────────┘            └─────────┘           └─────────┘           └─────────┘
```

---

### Workflow B: Running Agents (What perf-agents-bench Does)

**Goal:** Test how well AI agents can optimize code.

```bash
cd perf-agents-bench

# Step 1: Create a plan
# "Which commits should we test?"
python -m bench.cli plan tasks/vllm.yaml --out state/plan.json

# Step 2: Run agents
# "Let the AI try to optimize each task"
python -m bench.cli prepare tasks/vllm.yaml \
  --from-plan state/plan.json \
  --bench-cfg bench.yaml \
  --max-workers 1

# Step 3: Collect results
# "Gather all the agent's answers"
find state/runs -name prediction.jsonl -exec cat {} + > all_predictions.jsonl

# Step 4: Grade with GSO harness
# "Did the agent's changes actually make things faster?"
python ../gso/src/gso/harness/opt_at_k.py \
  --prediction_paths all_predictions.jsonl \
  --k 10
```

**Visual:**

```
          Plan                    Prepare                     Report
     ──────────────          ──────────────────          ──────────────
           │                        │                          │
           ▼                        ▼                          ▼
     ┌──────────┐             ┌──────────┐               ┌──────────┐
     │ 99       │             │ Agent    │               │ Opt@K    │
     │ commits  │──resolve───►│ works in │──collect─────►│ metrics  │
     │ to test  │             │ worktree │               │ (grades) │
     └──────────┘             └──────────┘               └──────────┘
                                   │
                                   ▼
                              Agent makes
                              changes, we
                              capture diff
```

---

### Workflow C: The OmniPerf-Bench Way (Current Approach)

**Goal:** Simplified pipeline using pre-extracted commits.

```bash
# Single command to process commits into dataset
python commit_to_dataset.py configs/experiments.yaml

# This does:
# 1. Reads commit JSONs from inputs/experiments/
# 2. Generates tests with LLM
# 3. Runs tests to measure speedup
# 4. Outputs data/vllm_dataset_with_test.jsonl
```

**Visual:**

```
┌───────────────────────────────────────────────────────────────────────┐
│                    OMNIPERF-BENCH SIMPLIFIED FLOW                     │
├───────────────────────────────────────────────────────────────────────┤
│                                                                       │
│   inputs/experiments/                    commit_to_dataset.py         │
│   └── commit_extractions_with_apis/           │                       │
│       ├── abc123.json  ─────────────────────► │                       │
│       ├── def456.json  ─────────────────────► │ ──► LLM generates     │
│       └── ghi789.json  ─────────────────────► │     tests             │
│                                               │                       │
│                                               ▼                       │
│                                         Run tests on                  │
│                                         base/head/main                │
│                                               │                       │
│                                               ▼                       │
│                                     data/vllm_dataset_with_test.jsonl │
│                                                                       │
└───────────────────────────────────────────────────────────────────────┘
```

---

## 6. Problems and Gaps

### Problem 1: Two Parallel Systems

**The Issue:**
```
GSO has:                          OmniPerf-Bench has:
├── Full commit extraction        ├── Pre-extracted commits (manual)
├── API mapping with RAG          ├── API mapping in JSON files
├── Sophisticated test gen        ├── Simpler test generation
├── Cloud execution (SkyPilot)    ├── Cloud execution (same)
└── Dataset building              └── Dataset building (different)
```

**Why This is Confusing:**
- Two ways to do the same thing
- Different code paths for similar operations
- Unclear which to use when

**The Gap:** Need to either:
1. Fully adopt GSO's pipeline, OR
2. Document clear differences and when to use each

---

### Problem 2: Evaluation Depends on GSO

**The Issue:**
```python
# In src/harness/opt_at_k.py
# We call GSO's code via subprocess:
cmd = [
    "uv", "run",
    "src/gso/harness/opt_at_k.py",  # <-- Depends on GSO
    ...
]
```

**Why This Matters:**
- Can't evaluate without GSO installed
- Version mismatches can break things
- Two systems need to stay in sync

**The Gap:** Need to either:
1. Vendor GSO's harness code, OR
2. Make GSO a proper dependency with version pinning

---

### Problem 3: Dataset Format Variations

**The Issue:**
```
GSO expects:                      OmniPerf-Bench produces:
{                                 {
  "instance_id": "...",             "instance_id": "...",
  "opt_commit": "...",              "head_commit": "...",      # Different name!
  "prob_script": "...",             "efficiency_test": "...",  # Different name!
  ...                               ...
}                                 }
```

**Why This Matters:**
- Converting between formats is error-prone
- Some fields might be lost in translation
- Hard to know which format to use

**The Gap:** Need a canonical schema (documented in `docs/dataset_schema.md`) and proper converters.

---

### Problem 4: Agent Isolation Varies

**The Issue:**
```
OpenHands:     Docker container (fully isolated)
TRAE:          Git worktree (partially isolated)
Codex CLI:     Git worktree (partially isolated)
```

**Why This Matters:**
- Different agents have different "cheating" opportunities
- Hard to compare results fairly
- Security implications

**The Gap:** Need consistent isolation policy across all agents.

---

### Problem 5: Resume Logic is Fragmented

**The Issue:**
```
perf-agents-bench --resume:
├── Only works within ONE run session
├── Doesn't check other run directories
└── Have to manually filter plan.json

GSO:
├── Different resume logic
└── Uses different state tracking
```

**Why This Matters:**
- Running 100 commits takes hours/days
- If it crashes, you might redo work
- Manual filtering is error-prone

**The Gap:** Need unified resume logic that:
1. Tracks ALL completed work across ALL runs
2. Automatically skips completed items
3. Works the same in all systems

---

### Problem 6: No Unified Entry Point

**The Issue:**
```
To create a dataset:        commit_to_dataset.py  OR  GSO's pipeline
To run agents:              perf-agents-bench CLI
To evaluate:                src/harness/opt_at_k.py  OR  GSO's harness
To analyze:                 Various scripts in docs/analysis/
```

**Why This Matters:**
- New users don't know where to start
- Easy to use wrong tool for the job
- Documentation scattered across files

**The Gap:** Need a single CLI or clear decision tree for all operations.

---

### Summary of Gaps

| Gap | Severity | Impact |
|-----|----------|--------|
| Two parallel systems | High | Confusion, maintenance burden |
| GSO dependency | Medium | Tight coupling, version issues |
| Format variations | Medium | Data loss, conversion errors |
| Inconsistent isolation | Medium | Unfair comparisons |
| Fragmented resume | High | Wasted compute, frustration |
| No unified entry | Medium | Steep learning curve |

---

## 7. How to Replicate GSO

### What GSO Has That You Need

```
┌─────────────────────────────────────────────────────────────────────┐
│                     GSO REPLICATION CHECKLIST                       │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  Component              GSO Has              You Have    Status     │
│  ─────────────────────────────────────────────────────────────────  │
│                                                                     │
│  1. Commit Extraction   LLM-powered          JSON files  Partial    │
│     (find perf commits) sophisticated        pre-made              │
│                         filtering                                   │
│                                                                     │
│  2. API Mapping         RAG + LLM            In JSON     Partial    │
│     (which APIs?)       analysis             files                  │
│                                                                     │
│  3. Test Generation     Multi-stage          Single LLM  Have it    │
│     (create tests)      with context         call                   │
│                                                                     │
│  4. Cloud Execution     SkyPilot             SkyPilot    Have it    │
│     (run on VMs)        (same code)          (shared)              │
│                                                                     │
│  5. Speedup Measurement Statistical          Basic       Partial    │
│     (how much faster?)  analysis             timing                 │
│                                                                     │
│  6. Dataset Building    Filtering +          Manual      Partial    │
│     (package it)        heuristics           JSONL                  │
│                                                                     │
│  7. Docker Evaluation   Full harness         Wraps GSO   Have it    │
│     (grade answers)                                                 │
│                                                                     │
│  8. Agent Execution     N/A (dataset only)   Full!       Better!    │
│     (run AI agents)                                                 │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### Option 1: Use GSO Directly

The simplest path to GSO-quality datasets:

```bash
# Install GSO
cd /path/to/OmniPerf-Bench/gso
uv venv && source .venv/bin/activate
uv pip install -e .

# Run GSO's pipeline
python src/gso/collect/analysis/commits.py my_config.yaml
python src/gso/collect/analysis/apis.py my_repo
python src/gso/collect/generate/generate.py my_config.yaml
python src/gso/collect/execute/execute.py --exp_id my_repo --machines 8
python src/gso/collect/build_dataset.py --exp_id my_repo
```

**Pros:** Full GSO quality, all features
**Cons:** Learning curve, need cloud setup

### Option 2: Enhance Current Pipeline

Improve OmniPerf-Bench's pipeline to match GSO:

```bash
# 1. Use better commit extraction
#    Replace manual JSONs with GSO's extraction:
python src/gso/collect/analysis/commits.py configs/vllm.yaml
# Output: Properly filtered performance commits

# 2. Improve test generation
#    Current: Single LLM call
#    GSO: Multi-stage with context retrieval
#    Action: Port gso/collect/generate/ logic

# 3. Add statistical analysis
#    Current: Mean timing only
#    GSO: Std dev, outlier detection, geometric mean
#    Action: Port gso/collect/execute/evaluate.py logic

# 4. Add filtering heuristics
#    Current: Include everything
#    GSO: Filter by speedup >= 1.2x, max tests, etc.
#    Action: Port gso/collect/build_dataset.py logic
```

### Option 3: Hybrid Approach (Recommended)

Use the best of both:

```
┌─────────────────────────────────────────────────────────────────────┐
│                    RECOMMENDED HYBRID APPROACH                      │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│   Dataset Creation:    Use GSO's collect/ pipeline                  │
│   Agent Execution:     Use perf-agents-bench (you have more agents!)│
│   Evaluation:          Use GSO's harness/                           │
│                                                                     │
│   Why?                                                              │
│   • GSO is polished for dataset creation                            │
│   • perf-agents-bench has better agent support                      │
│   • Both produce compatible formats                                 │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### Configuration for Replication

Create `configs/my_experiment.yaml`:

```yaml
# Experiment configuration
exp_id: "my_benchmark"
repo_url: "https://github.com/my-org/my-repo.git"
py_version: "3.12"
target_commit: "main"

# Installation
install_commands:
  - "pip install -e ."
  - "pip install pytest"

# LLM settings
llm_provider: "openai"  # or "anthropic", "bedrock"
llm_model: "gpt-4o"
llm_temperature: 0.1

# Filtering
min_speedup: 1.2        # Only include if 1.2x+ faster
max_tests_per_api: 20   # Limit tests per API
min_lines_changed: 5    # Skip trivial changes

# Output
dataset_name: "my_benchmark_dataset"
output_dir: "data/"
```

---

## 8. Quick Reference

### Commands Cheat Sheet

```bash
# ═══════════════════════════════════════════════════════════
#                    DATASET CREATION
# ═══════════════════════════════════════════════════════════

# Simple way (OmniPerf-Bench)
python commit_to_dataset.py configs/experiments.yaml

# Full way (GSO pipeline)
cd gso
python src/gso/collect/analysis/commits.py config.yaml   # Extract
python src/gso/collect/analysis/apis.py repo             # Map APIs
python src/gso/collect/generate/generate.py config.yaml  # Generate tests
python src/gso/collect/execute/execute.py --exp_id repo  # Execute
python src/gso/collect/build_dataset.py --exp_id repo    # Build

# ═══════════════════════════════════════════════════════════
#                    AGENT EXECUTION
# ═══════════════════════════════════════════════════════════

cd perf-agents-bench

# Plan
python -m bench.cli plan tasks/vllm.yaml --out state/plan.json

# Run agents (TRAE)
export TRAE_PYTHON=/path/to/python
export TRAE_CONFIG=/path/to/trae_config.yaml
python -m bench.cli prepare tasks/vllm.yaml \
  --from-plan state/plan.json \
  --bench-cfg bench.yaml

# Run agents (Codex CLI)
python -m bench.cli prepare tasks/vllm.yaml \
  --from-plan state/plan.json \
  --bench-cfg bench_codex.yaml

# Report
python -m bench.cli report state/runs/<run_id>

# ═══════════════════════════════════════════════════════════
#                     EVALUATION
# ═══════════════════════════════════════════════════════════

# Collect predictions
find perf-agents-bench/state/runs -name prediction.jsonl \
  -exec cat {} + > all_predictions.jsonl

# Grade with GSO harness
cd gso
uv run src/gso/harness/opt_at_k.py \
  --prediction_paths ../all_predictions.jsonl \
  --k 10 \
  --timeout 3600

# ═══════════════════════════════════════════════════════════
#                    DIAGNOSTICS
# ═══════════════════════════════════════════════════════════

# Check completed commits
grep -r '"status": "success"' perf-agents-bench/state/runs/

# Count by status
cd perf-agents-bench
for status in success error abandoned; do
  echo "$status: $(grep -r "\"status\": \"$status\"" state/runs/ | wc -l)"
done

# Total cost
find state/runs -name journal.json \
  -exec jq -r '.token_usage.total_cost_usd // 0' {} + | \
  awk '{sum+=$1} END {print "Total: $" sum}'
```

### File Locations

| What | Where |
|------|-------|
| Dataset output | `data/*.jsonl` |
| Commit extractions | `inputs/experiments/commit_extractions_with_apis/` |
| Test generators | `inputs/experiments/generated_test_generators_v4/` |
| Agent plans | `perf-agents-bench/state/plan.json` |
| Agent runs | `perf-agents-bench/state/runs/<run_id>/` |
| Agent journals | `perf-agents-bench/state/runs/<run_id>/<item_id>/journal.json` |
| Agent patches | `perf-agents-bench/state/runs/<run_id>/<item_id>/model_patch.diff` |
| GSO harness | `gso/src/gso/harness/` |
| Experiment configs | `configs/*.yaml` |

### Key Metrics

| Metric | What It Means | Good Value |
|--------|---------------|------------|
| Opt@1 | Success on first try | >10% |
| Opt@5 | Success in 5 tries | >30% |
| Opt@10 | Success in 10 tries | >50% |
| Speedup | How much faster | >1.2x |
| human_performance | Expert's speedup | Reference |

---

## Glossary

| Term | Definition |
|------|------------|
| **GSO** | Global Software Optimization - the comprehensive benchmark framework |
| **perf-agents-bench** | Agent execution framework for running OpenHands/TRAE/Codex |
| **Opt@K** | Optimization at K attempts - success rate metric |
| **Worktree** | Git feature for isolated working directories |
| **Journal** | Metadata file about agent execution |
| **Prediction** | Agent's submitted patch in GSO format |
| **Speedup** | Ratio of old_time/new_time (>1 means faster) |
| **Base commit** | The commit BEFORE optimization |
| **Head commit** | The commit WITH optimization |
| **Ground truth** | The expert developer's actual solution |

---

## Next Steps

1. **If you want to create a new dataset:** Start with GSO's pipeline
2. **If you want to test agents:** Use perf-agents-bench
3. **If you want to grade results:** Use GSO's harness
4. **If you find bugs:** Check the [Problems and Gaps](#6-problems-and-gaps) section

---

*Last updated: December 2025*
*For questions: Check CLAUDE.md or open an issue*
