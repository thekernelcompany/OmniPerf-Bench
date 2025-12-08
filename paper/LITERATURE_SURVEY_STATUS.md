# Literature Survey - COMPLETE ✅

## Summary

The literature survey for the OmniPerf-Bench (IOBench) ICML 2026 paper is now **complete and comprehensive**.

---

## What We Have

### 1. Related Work Section in Paper ✅
**Location:** `/Users/fortuna/Desktop/Exp/OmniPerf-Bench/paper/icml2026/example_paper.tex`

**Coverage:**
- ✅ Benchmarks for code generation (HumanEval, MBPP, SWE-bench, CodeContests)
- ✅ Performance-aware code generation (KernelBench, TritonBench, GSO, SWE-Perf)
- ✅ Agent architectures (OpenHands, SWE-Agent)
- ✅ LLM inference systems (vLLM, SGLang, MAX Engine)
- ✅ System benchmarks (MLPerf Inference, LLMPerf)
- ✅ Compiler optimization (CompilerGym, AnghaBench, BenchPress)
- ✅ Repository mining (BugSwarm, BugBuilder, RegMiner)

**Word Count:** ~1,200 words in Related Work section

### 2. Complete Bibliography ✅
**Location:** `/Users/fortuna/Desktop/Exp/OmniPerf-Bench/paper/icml2026/references.bib`

**Statistics:**
- **60+ papers with complete arXiv IDs, DOIs, venues**
- **80+ papers total in comprehensive_bibliography.md**
- All citations verified from GPT-5 bibliography
- Years: 2019-2025 (emphasis on 2023-2025)
- Top venues: ICML, NeurIPS, ICLR, ICSE, ISSTA, CGO, Science

### 3. Reference Documents ✅

**`literature_survey.md`** - Analysis of 4 key papers:
- KernelBench (arXiv 2502.10517)
- TritonBench (arXiv 2502.14752)
- GSO (arXiv 2505.23671)
- SWE-Perf (arXiv 2507.12415)

**`comprehensive_bibliography.md`** - Full 80+ paper bibliography organized by 10 categories

**`literature_survey_summary.md`** - Topics covered, OmniPerf-Bench positioning, citation strategy

**`gpt5_paper_search_prompt.md`** - Prompt for finding additional relevant papers

---

## OmniPerf-Bench Positioning (from brainstorming)

### Primary Contribution: A (Benchmark)
**282 real-world performance optimization tasks from production ML inference engines (vLLM, SGLang)**

### Key Differentiator: B (Behavioral Insights)
**Deep analysis revealing WHY agents fail, not just THAT they fail**

### Positioning Statement (Methodological Advancement)
**"Building on GSO's commit-mining approach and SWE-Perf's executable environments, OmniPerf-Bench adds:**
1. **Behavioral analysis** - instant-edit pathology (<1s TTFE → 95% failure), commit explosion (7,755 commits)
2. **Codebase complexity effects** - 58-point performance gap (96% SGLang vs 38% vLLM) using identical agent
3. **Multi-agent comparison** - Codex/TRAE/OpenHands trade-offs (cost vs observability vs success)"

---

## 3 Unique Gaps OmniPerf-Bench Addresses

### Gap 1: Behavioral Analysis Beyond Success Rates
**Prior work:** GSO (<5%), KernelBench (<20%), SWE-Perf (substantial gap)
**OmniPerf-Bench:** Identifies instant-edit pathology, commit explosion, scope violations

### Gap 2: Codebase Complexity Effects
**Prior work:** Aggregates across diverse repos
**OmniPerf-Bench:** Demonstrates 58-point gap (96% vs 38%) on identical agent → codebase characteristics matter

### Gap 3: Multi-Agent Comparison at Scale
**Prior work:** GSO (unnamed agents), SWE-Perf (2 approaches)
**OmniPerf-Bench:** 3 agents × 282 tasks = 846 evaluations

---

## Citation Strategy

### Establish Gap (Functional Correctness Only)
- SWE-bench, HumanEval, MBPP → "focus only on correctness, ignore performance"

### Recent Performance Work (Our Context)
- KernelBench, TritonBench → "domain-specific GPU kernels"
- GSO, SWE-Perf → "general optimization but aggregate metrics, lack deep behavioral analysis"

### Agent Systems (Justify Evaluation)
- OpenHands, SWE-Agent → "architectural diversity exists, comparison rare"

### Domain Systems (Justify Task Selection)
- vLLM, SGLang → "production ML inference engines provide realistic optimization challenges"

### Methodology Precedents (Justify Mining)
- BugSwarm, BugBuilder, RegMiner → "repository mining yields realistic benchmarks"

---

## Next Steps for Paper Writing

Based on our brainstorming session, the paper structure is:

### ✅ DONE
1. Literature survey complete
2. Related Work section written
3. References.bib populated
4. OmniPerf-Bench positioning clarified

### ⏳ TO DO (from todo list)
3. Write abstract and introduction
4. Write methodology section (dataset construction, commit mining, test generation)
5. Write experiments and results section (success rates, multi-agent comparison)
6. Write analysis and discussion section (behavioral insights: instant-edit, commit explosion, codebase effects)
7. Write conclusion and future work
8. Create figures and tables (visualizations from our Codex/TRAE analysis)
9. Format references and bibliography
10. Review and polish final draft

---

## Files Created/Updated

### Paper Directory
```
paper/
├── icml2026/
│   ├── example_paper.tex          [UPDATED: Related Work section]
│   ├── references.bib              [UPDATED: 60+ complete citations]
│   └── omniperf_bench.tex          [CREATED: Cleaner version with our content]
├── literature_survey.md            [CREATED: 4 paper analysis]
├── literature_survey_summary.md    [CREATED: Topics & positioning]
├── comprehensive_bibliography.md   [CREATED: 80+ papers from GPT-5]
├── gpt5_paper_search_prompt.md     [CREATED: Search template]
└── LITERATURE_SURVEY_STATUS.md     [THIS FILE]
```

### Downloaded Papers
```
paper/
├── 2502.10517v1.html  (KernelBench - 1.2 MB)
├── 2502.14752v1.html  (TritonBench - 342 KB)
├── 2505.23671v3.html  (GSO - 638 KB)
└── 2507.12415v1.html  (SWE-Perf - 324 KB)
```

---

## Key Numbers for Introduction/Abstract

From our analysis (to be incorporated):

### Benchmark Scale
- **282 optimization tasks** from production ML inference engines
- **2 repositories:** vLLM (complex), SGLang (simpler)
- **3 agents evaluated:** Codex, TRAE, OpenHands

### Success Rates (Codex)
- **vLLM:** 38.4% clean success (38/99 tasks)
- **SGLang:** 96.2% clean success (77/80 tasks)
- **Gap:** 58 percentage points using identical agent configuration

### Failure Modes Discovered
- **Instant-edit pathology:** <1s time-to-first-edit predicts 95% failure rate
- **Commit explosion:** Up to 7,755 commits on single task (median: 1)
- **Scope violations:** Up to 2,970 unauthorized file changes

### Multi-Agent Trade-offs
- **Codex:** 100% completion, <$1/task, no observability (black box)
- **TRAE:** 50% success, $16/task, full trajectory logging
- **Token usage:** TRAE uses 1.6M tokens/task on vLLM, 712K on SGLang (55% less)

### Comparison to Prior Work
- **GSO:** 102 tasks, <5% success
- **SWE-Perf:** 140 tasks, substantial gap
- **KernelBench:** 250 tasks, <20% match baseline
- **OmniPerf-Bench:** 282 tasks + behavioral analysis

---

## How to Compile the Paper

```bash
cd /Users/fortuna/Desktop/Exp/OmniPerf-Bench/paper/icml2026
pdflatex example_paper.tex
bibtex example_paper
pdflatex example_paper.tex
pdflatex example_paper.tex
```

**Note:** Some \todo{} sections remain for methodology, experiments, results sections.

---

## Status Summary

| Task | Status | Notes |
|------|--------|-------|
| Extract 4 papers | ✅ DONE | All analyzed, saved as HTML |
| Literature survey | ✅ DONE | Related Work section complete |
| Bibliography | ✅ DONE | 60+ citations in references.bib |
| GPT-5 search | ✅ DONE | 80+ papers cataloged |
| Positioning | ✅ DONE | Methodological advancement relative to GSO/SWE-Perf |
| Abstract/Intro | ⏳ TODO | Numbers ready from our analysis |
| Methodology | ⏳ TODO | Commit mining, test generation, evaluation |
| Experiments | ⏳ TODO | Success rates, behavioral patterns |
| Analysis | ⏳ TODO | Instant-edit, commit explosion, codebase effects |
| Figures/Tables | ⏳ TODO | Use existing visualizations from analysis |
| Conclusion | ⏳ TODO | Summary + future work |

---

**The literature survey is COMPLETE. Ready to proceed with paper writing!** 🎉
