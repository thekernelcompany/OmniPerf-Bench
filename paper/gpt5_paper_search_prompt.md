# GPT-5 Prompt for Finding Relevant Papers for OmniPerf-Bench

## Context
I'm writing a paper for ICML 2026 on **OmniPerf-Bench** (working title: IOBench), a benchmark for evaluating AI coding agents on real-world performance optimization tasks. The benchmark contains 282 optimization tasks from production ML inference engines (vLLM, SGLang), and we evaluate multiple agents (Codex, TRAE, OpenHands) on their ability to optimize code performance.

## Search Request

Please search for and provide the most relevant papers in the following categories. For each paper, provide: title, authors, venue, year, arXiv ID (if applicable), and a 2-3 sentence summary of relevance.

### Category 1: AI Coding Agent Benchmarks (Functional Correctness)

Find papers on benchmarks that evaluate AI/LLM coding capabilities on functional correctness:
- **SWE-bench** - Repository-level bug fixing from GitHub issues
- **HumanEval** - Function-level programming tasks
- **MBPP** (Mostly Basic Programming Problems)
- **CodeContests** - Competitive programming challenges
- **APPS** - Automated Programming Progress Standard
- **ClassEval** - Class-level code generation
- **Any other major code generation benchmarks** from 2020-2025

**Search queries:**
- "SWE-bench software engineering benchmark"
- "HumanEval code generation benchmark"
- "MBPP programming benchmark"
- "code generation benchmark evaluation LLM"

### Category 2: Performance-Aware Code Generation

Find papers specifically about generating or optimizing code for PERFORMANCE (not just correctness):
- **KernelBench** - GPU kernel generation (already have: arXiv 2502.10517)
- **TritonBench** - Triton operator generation (already have: arXiv 2502.14752)
- **Polybench** or other performance benchmarks
- **Any work on LLMs generating efficient/fast code**
- **GPU kernel optimization with LLMs**
- **CUDA/Triton code generation**

**Search queries:**
- "GPU kernel generation language models"
- "performance optimization code generation"
- "efficient code generation benchmark"
- "Triton CUDA kernel LLM"

### Category 3: Software Performance Optimization Benchmarks

Find papers on benchmarking agents' ability to OPTIMIZE existing code:
- **GSO** - Software optimization tasks (already have: arXiv 2505.23671)
- **SWE-Perf** - Performance optimization from GitHub PRs (already have: arXiv 2507.12415)
- **Performance bug detection and fixing**
- **Code optimization datasets**
- **Compiler optimization benchmarks**

**Search queries:**
- "software performance optimization benchmark"
- "code optimization evaluation"
- "performance bug fixing dataset"
- "SWE-Agent performance optimization"

### Category 4: Agent Architectures and Frameworks

Find papers on AI agent systems for coding:
- **OpenHands/OpenDevin** - Open-source coding agent framework
- **SWE-Agent** - Agent-computer interface for software engineering
- **AutoCodeRover** - Autonomous code fixing
- **Aider** - AI pair programming
- **Mentat** - AI coding assistant
- **MetaGPT** - Multi-agent framework
- **AgentBench** - General agent evaluation

**Search queries:**
- "OpenHands coding agent"
- "SWE-Agent autonomous software engineering"
- "AI coding agent framework"
- "autonomous code repair agent"

### Category 5: LLM Inference Systems (Domain Context)

Find papers on the SYSTEMS we're optimizing (our benchmark domain):
- **vLLM** - PagedAttention, high-throughput LLM serving
- **SGLang** - Structured generation, DSL for LLM serving
- **TensorRT-LLM** - NVIDIA inference optimization
- **FlashAttention** - Efficient attention mechanisms
- **Any survey on LLM inference optimization**

**Search queries:**
- "vLLM PagedAttention inference"
- "SGLang structured generation"
- "LLM inference optimization survey"
- "efficient transformer serving"

### Category 6: Repository Mining and Commit Analysis

Find papers on extracting benchmarks/datasets from git repositories:
- **BugSwarm** - Mining CI failures from Travis/GitHub Actions
- **BugBuilder** - Extracting bug fixes from commits
- **RegMiner** - Regression bug mining
- **Defects4J** - Real bugs from Java projects
- **Any work on mining performance commits**

**Search queries:**
- "BugSwarm continuous integration mining"
- "commit history mining benchmark"
- "software repository mining dataset"
- "performance regression detection"

### Category 7: Evaluation Methodologies for Code Agents

Find papers on HOW to evaluate coding agents:
- **Metrics for code quality** (beyond pass@k)
- **Agent trajectory analysis**
- **Tool use patterns in coding agents**
- **Behavioral analysis of AI agents**
- **Multi-agent comparison methodologies**

**Search queries:**
- "evaluating AI coding agents"
- "code quality metrics LLM"
- "agent trajectory analysis"
- "tool use patterns language models"

### Category 8: Compiler Optimization and Learning

Find papers on using ML for compiler optimization (related work):
- **CompilerGym** - RL environments for compiler tasks
- **AnghaBench** - Large corpus of C functions for optimization
- **BenchPress** - Synthesizing compiler benchmarks
- **Neural program optimization**

**Search queries:**
- "CompilerGym reinforcement learning"
- "machine learning compiler optimization"
- "neural program optimization"
- "learned optimizers code"

### Category 9: Recent Work on Code LLMs (2024-2025)

Find the most recent papers on state-of-the-art code models:
- **GPT-4/GPT-4o** code capabilities
- **Claude 3/Claude 3.5 Sonnet** for coding
- **DeepSeek Coder**
- **CodeLlama**, **StarCoder**, **CodeGen**
- **Qwen Coder**
- **Any code-specific model papers from 2024-2025**

**Search queries:**
- "GPT-4 code generation evaluation"
- "DeepSeek Coder benchmark"
- "code language models 2024"
- "Claude coding capabilities"

### Category 10: Performance Analysis Tools and Profiling

Find papers on automated performance analysis (relevant to our methodology):
- **Automated bottleneck detection**
- **Performance profiling for Python/ML code**
- **Static/dynamic performance analysis**
- **Performance bug detection**

**Search queries:**
- "automated performance analysis"
- "bottleneck detection machine learning"
- "performance profiling tools"
- "performance bug detection"

## Output Format

For each paper found, please provide:

```
### Paper Title
**Authors:** [Full author list]
**Venue:** [Conference/Journal, Year]
**arXiv ID:** [if available]
**DOI:** [if available]
**Relevance:** [2-3 sentences explaining why this paper is relevant to OmniPerf-Bench]
**Key metrics/results:** [If applicable, main quantitative findings]
```

## Priority

**Highest priority (must find):**
1. SWE-bench paper
2. HumanEval paper
3. MBPP paper
4. OpenHands/OpenDevin
5. SWE-Agent
6. vLLM system paper
7. SGLang system paper
8. CompilerGym
9. BugSwarm

**Also important:**
- Any 2024-2025 papers on code optimization with LLMs
- Recent surveys on AI coding agents
- Recent surveys on LLM inference optimization

## Additional Instructions

- Prioritize papers from top-tier venues (ICML, NeurIPS, ICLR, ACL, EMNLP, PLDI, OSDI, SOSP, MLSys)
- Include recent arXiv preprints (2024-2025) even if not published
- For each category, aim for 3-5 most relevant papers
- Include both seminal works and recent state-of-the-art
- If a paper has been updated, provide the latest version

Thank you!
