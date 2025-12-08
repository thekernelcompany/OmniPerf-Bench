# Literature Survey for OmniPerf-Bench Paper

## Paper Summaries

### 1. KernelBench: Can LLMs Write Efficient GPU Kernels? (arXiv 2502.10517)

**Authors:** Anne Ouyang, Simon Guo, Simran Arora, Alex L. Zhang, William Hu, Christopher Ré, Azalia Mirhoseini

**Venue:** arXiv preprint 2025

**Main Contributions:**
- First open-source evaluation framework for GPU kernel generation
- Introduces `fast_p` metric: percentage of functionally correct kernels exceeding speedup thresholds
- 250 carefully curated PyTorch ML workloads
- Evaluates iterative refinement with execution and profiling feedback
- Establishes challenging real-world engineering benchmark

**Methodology:**
- Benchmark suite: 250 PyTorch workloads
- Evaluation: Functional correctness + performance (speedup vs baseline)
- Iterative refinement using execution feedback and profiling data
- Test-time methods for kernel optimization

**Key Results:**
- Frontier reasoning models: <20% match PyTorch baseline performance
- Iterative refinement improves results
- Performance degrades with increasing speedup threshold requirements

**Models Evaluated:** State-of-the-art reasoning models (specific names not disclosed in abstract)

**Relation to OmniPerf-Bench:**
- Both focus on performance optimization beyond functional correctness
- KernelBench: GPU kernels specifically, OmniPerf-Bench: general code optimization
- KernelBench: Synthetic workloads, OmniPerf-Bench: Real commit history
- Both use speedup metrics for evaluation

---

### 2. TritonBench: Benchmarking LLM Capabilities for Generating Triton Operators (arXiv 2502.14752)

**Authors:** Jianling Li, Shangzhan Li, Zhenye Gao, Qi Shi, Yuxuan Li, Zefan Wang, Jiacheng Huang, Haojie Wang, Jianrong Wang, Xu Han, Zhiyuan Liu, Maosong Sun

**Venue:** arXiv preprint 2025

**Main Contributions:**
- First comprehensive benchmark for Triton operator generation
- Dual evaluation channels: 184 real-world operators from GitHub + PyTorch-aligned operators
- Functional correctness AND efficiency performance evaluation on industry GPUs
- Systematic evaluation framework for high-performance code generation
- Reveals significant gaps in current LLM capabilities

**Methodology:**
- Two-channel approach:
  1. Real-world operators from GitHub repositories
  2. PyTorch-aligned operator interfaces
- Performance profiling on production-grade GPUs
- Industry-standard evaluation environment

**Key Results:**
- Current state-of-the-art code LLMs struggle with efficient Triton generation
- Quantitative metrics not disclosed in abstract

**Models Evaluated:** State-of-the-art code LLMs (names not specified)

**Relation to OmniPerf-Bench:**
- Both emphasize efficiency beyond correctness
- TritonBench: Domain-specific (GPU kernels), OmniPerf-Bench: General optimization
- TritonBench: Generation from scratch, OmniPerf-Bench: Optimization of existing code
- Both use real-world benchmarks

---

### 3. GSO: Challenging Software Optimization Tasks for Evaluating SWE-Agents (arXiv 2505.23671)

**Authors:** Manish Shetty, Naman Jain, Jinjian Liu, Vijay Kethanaboyina, Koushik Sen, Ion Stoica

**Venue:** arXiv preprint 2025

**Main Contributions:**
- First benchmark for evaluating SWE-Agents on performance optimization
- Automated pipeline generating performance tests from commit history
- 102 difficult optimization tasks across 10 codebases
- Diverse programming languages and application domains
- Identifies key failure patterns (low-level languages, bottleneck localization)

**Methodology:**
- Automated pipeline analyzing repository commit histories
- Performance test generation and execution
- Agents receive: codebase + performance specification
- Evaluation: Runtime improvement vs expert developer baselines

**Key Results:**
- Leading SWE-Agents: <5% success rate
- Minimal gains from inference-time scaling
- Struggles with low-level languages and bottleneck identification

**Models Evaluated:** Leading SWE-Agents (specific names not disclosed)

**Relation to OmniPerf-Bench:**
- **Very close alignment** - both use commit history for benchmark generation
- GSO: 102 tasks across 10 repos, OmniPerf-Bench: 282 tasks (vLLM/SGLang focus)
- GSO: <5% success, OmniPerf-Bench: Codex 38-96% (codebase-dependent)
- Both evaluate real-world optimization capabilities
- **Key difference:** OmniPerf-Bench includes multi-agent comparison (Codex/TRAE) + detailed behavioral analysis

---

### 4. SWE-Perf: Can LLMs Optimize Code Performance on Real-World Repositories? (arXiv 2507.12415)

**Authors:** Xinyi He, Qian Liu, Mingzhe Du, Lin Yan, Zhijie Fan, Yiming Huang, Zejian Yuan, Zejun Ma

**Venue:** arXiv preprint 2025

**Main Contributions:**
- First systematic evaluation of LLMs on code performance optimization
- 140 benchmark instances from actual GitHub performance-enhancing PRs
- File-level AND repository-level evaluation approaches
- Establishes research foundation for LLM-based performance enhancement
- Reveals substantial capability gaps vs expert developers

**Methodology:**
- 140 instances from real GitHub performance PRs
- Each instance contains:
  - Relevant codebase
  - Target functions
  - Performance tests
  - Expert-authored patches
  - Executable environments
- Evaluates Agentless (file-level) and OpenHands (repo-level)

**Key Results:**
- Substantial capability gap between LLMs and experts
- Specific success rates not disclosed in abstract

**Models Evaluated:**
- Agentless (file-level approach)
- OpenHands (repository-level approach)

**Relation to OmniPerf-Bench:**
- Both use real-world GitHub optimization commits
- SWE-Perf: 140 instances, OmniPerf-Bench: 282 instances
- SWE-Perf includes OpenHands (which OmniPerf-Bench also evaluates)
- **Key difference:** OmniPerf-Bench adds TRAE, Codex comparison + behavioral analysis (commit patterns, violations)
- Both provide executable environments for performance testing

---

## Literature Survey Topic Structure

Based on these 4 papers and the broader field, the Related Work section should cover:

### 1. AI Coding Agent Benchmarks (Functional Correctness)
- **SWE-bench** (Jimenez et al., 2023): Repository-level bug fixing, 2,294 GitHub issues
- **HumanEval** (Chen et al., 2021): 164 function-level programming tasks
- **MBPP** (Austin et al., 2021): 974 basic programming problems
- **CodeContests** (Li et al., 2022): Competitive programming challenges
- **Gap:** Focus on functional correctness, not performance optimization

### 2. Performance-Aware Code Generation
- **KernelBench** (Ouyang et al., 2025): GPU kernel generation, 250 PyTorch workloads, <20% baseline match
- **TritonBench** (Li et al., 2025): Triton operator generation, 184 real-world operators
- **Polybench** (Prior work on performance benchmarks)
- **Gap:** Domain-specific (GPU kernels), not general code optimization

### 3. Software Performance Optimization Benchmarks
- **GSO** (Shetty et al., 2025): 102 optimization tasks across 10 repos, <5% agent success
- **SWE-Perf** (He et al., 2025): 140 GitHub PR-based optimization tasks, evaluates Agentless/OpenHands
- **Gap:** Limited multi-agent comparison, no behavioral analysis of failure modes

### 4. Agent Evaluation Methodologies
- **AgentBench** (Liu et al., 2023): Multi-dimensional agent evaluation
- **WebArena** (Zhou et al., 2023): Web-based agent tasks
- **SWE-Agent** (Yang et al., 2024): Repository-level editing agents
- **Gap:** Not specialized for performance optimization

### 5. What OmniPerf-Bench Uniquely Provides

**Positioning Statement:**
"While recent benchmarks have begun exploring performance optimization (GSO, SWE-Perf, KernelBench), OmniPerf-Bench makes three unique contributions:

1. **Multi-Agent Behavioral Analysis**: Unlike prior work reporting only success rates, we provide deep behavioral analysis (commit patterns, violation trajectories, time-to-first-edit) revealing *why* agents fail, not just *that* they fail.

2. **Codebase Complexity Effects**: We demonstrate that codebase characteristics dramatically affect agent performance (96% clean success on SGLang vs 38% on vLLM using identical agent), a variable unexplored in prior benchmarks.

3. **Large-Scale Real-World Dataset**: 282 optimization tasks from production ML inference engines (vLLM/SGLang), providing domain-relevant challenges for high-performance computing applications.

4. **Agent Comparison**: Direct comparison of three distinct agent architectures (Codex, TRAE, OpenHands) on identical tasks, revealing architectural trade-offs (Codex: 100% task completion, <$1/task vs TRAE: 50% success, $16/task)."

---

## Related Work Section Draft (LaTeX)

```latex
\section{Related Work}

\subsection{Benchmarks for Code Generation}

The development of large language models for code has spurred extensive benchmark creation. Early work focused on functional correctness: \textbf{HumanEval}~\cite{chen2021evaluating} introduced 164 handwritten programming problems, while \textbf{MBPP}~\cite{austin2021program} provided 974 basic Python tasks. \textbf{SWE-bench}~\cite{jimenez2023swe} scaled to repository-level evaluation with 2,294 real-world GitHub issues requiring multi-file edits.

However, these benchmarks measure only whether code \emph{works}, not whether it performs \emph{well}. A functionally correct solution may be orders of magnitude slower than expert-written code—a critical gap for production systems.

\subsection{Performance-Aware Code Generation}

Recent work has begun addressing performance optimization. \textbf{KernelBench}~\cite{ouyang2025kernelbench} evaluates LLMs on GPU kernel generation across 250 PyTorch workloads, finding that frontier models match baseline performance in less than 20\% of cases. \textbf{TritonBench}~\cite{li2025tritonbench} focuses on Triton operator generation with 184 real-world operators, revealing struggles in producing efficient low-level code.

These benchmarks target domain-specific optimization (GPU kernels), whereas general-purpose performance optimization remains underexplored.

\subsection{Software Performance Optimization}

Most closely related to our work, \textbf{GSO}~\cite{shetty2025gso} and \textbf{SWE-Perf}~\cite{he2025sweperf} evaluate agents on general code optimization tasks. GSO curates 102 tasks across 10 repositories using an automated commit history pipeline, reporting <5\% success for leading agents. SWE-Perf provides 140 instances from GitHub performance-enhancing pull requests, evaluating file-level (Agentless) and repository-level (OpenHands) approaches.

While these establish the difficulty of optimization tasks, they report only aggregate success rates. Our work extends this by providing \emph{behavioral analysis}—why do agents fail? What patterns predict success?

\subsection{Agent Architectures}

Various agent architectures have emerged for code tasks. \textbf{OpenHands}~\cite{openhands2024} (formerly OpenDevin) provides an open-source framework for autonomous coding agents. \textbf{TRAE}~\cite{trae2024} introduces tool-augmented reasoning with execution feedback. Commercial systems like GitHub Copilot Workspace and Cursor explore agentic workflows.

However, comparative evaluation across agent architectures on identical tasks is rare, making it difficult to assess architectural trade-offs.

\subsection{OmniPerf-Bench Contributions}

OmniPerf-Bench addresses three gaps in existing benchmarks:

\textbf{(1) Behavioral Analysis Beyond Success Rates.} We analyze agent trajectories to identify failure modes: instant-edit pathology (<1s time-to-first-edit predicts 95\% failure), commit explosion (7,755 commits on single task), and scope violations (2,970 unauthorized file changes). This reveals \emph{why} agents fail, enabling targeted improvements.

\textbf{(2) Codebase Complexity Effects.} We demonstrate dramatic performance variance across codebases: Codex achieves 96\% clean success on SGLang but only 38\% on vLLM—a 58-point gap using identical agent configuration. This suggests codebase characteristics may be as important as agent architecture.

\textbf{(3) Multi-Agent Comparison.} We evaluate three agents (Codex, TRAE, OpenHands) on 282 identical tasks, revealing trade-offs: Codex completes 100\% of tasks at <\$1/task but lacks transparency; TRAE provides full observability but costs \$16/task with 50\% success rate.

Our dataset of 282 optimization tasks from production ML inference engines (vLLM, SGLang) provides domain-relevant challenges for high-performance computing applications, complementing existing benchmarks' focus on web frameworks and general utilities.
```

---

## Key Citations Needed

Based on the 4 papers + general knowledge, we need BibTeX entries for:

**From the 4 papers (already have arXiv IDs):**
1. ouyang2025kernelbench (2502.10517)
2. li2025tritonbench (2502.14752)
3. shetty2025gso (2505.23671)
4. he2025sweperf (2507.12415)

**Standard code benchmarks (need to find):**
5. chen2021evaluating (HumanEval)
6. austin2021program (MBPP)
7. jimenez2023swe (SWE-bench)
8. li2022codecontests (CodeContests)

**Agent frameworks (need to find):**
9. openhands2024 (OpenHands/OpenDevin)
10. trae2024 (if exists as publication)

**Additional potentially relevant:**
11. agentbench (Liu et al., 2023)
12. webarena (Zhou et al., 2023)
