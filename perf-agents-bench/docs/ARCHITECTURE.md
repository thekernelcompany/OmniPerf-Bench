# perf-agents-bench Architecture

## System Overview

```mermaid
graph TB
    subgraph "Input Layer"
        TC[tasks/*.yaml<br/>Task Config]
        CC[commits.txt<br/>Commit Pairs]
        BC[bench.yaml<br/>Bench Config]
        ENV[.env<br/>Credentials]
    end

    subgraph "CLI Commands"
        PLAN[bench.cli plan]
        PREP[bench.cli prepare]
        REP[bench.cli report]
        VAL[bench.cli validate]
        DOC[bench.cli doctor]
    end

    subgraph "Core Components"
        MP[MatrixPlanner<br/>Resolves commits]
        PE[PrepareExecutor<br/>Runs agents]
        RM[RepoManager<br/>Git operations]
        JW[JournalWriter<br/>Logs results]
    end

    subgraph "Agent Layer"
        OH[OpenHands Agent<br/>via uvx/docker]
        AB[Agent Base<br/>Abstract]
    end

    subgraph "Storage"
        PLANF[state/plan.json]
        RUNS[state/runs/<run_id>/]
        WT[.work/worktrees/]
    end

    TC --> PLAN
    CC --> PLAN
    PLAN --> MP
    MP --> PLANF

    PLANF --> PREP
    BC --> PREP
    ENV --> PREP
    PREP --> PE
    PE --> RM
    PE --> OH
    PE --> JW
    RM --> WT
    JW --> RUNS

    RUNS --> REP
    REP --> JSON[JSON Report]
```

## Detailed Flow Diagram

```mermaid
sequenceDiagram
    participant User
    participant CLI
    participant Planner
    participant Git
    participant Executor
    participant OpenHands
    participant Journal

    User->>CLI: plan tasks/vllm.yaml
    CLI->>Planner: build_plan()
    Planner->>Git: resolve commits
    Git-->>Planner: (human, pre) pairs
    Planner-->>CLI: plan.json
    CLI-->>User: ✓ Plan created

    User->>CLI: prepare --from-plan plan.json
    CLI->>Executor: execute()
    
    loop For each commit pair
        Executor->>Git: create_worktree(pre_commit)
        Git-->>Executor: worktree path
        
        Executor->>Journal: write_prompt()
        Executor->>Executor: create task.txt
        
        alt Container Mode
            Executor->>OpenHands: docker run openhands
        else Host Mode
            Executor->>OpenHands: uvx python -m openhands.core.main
        end
        
        OpenHands->>OpenHands: Read task.txt
        OpenHands->>OpenHands: Modify code
        OpenHands->>Git: git commit changes
        OpenHands-->>Executor: stdout/stderr
        
        Executor->>Git: get_changed_files()
        Executor->>Executor: validate targets
        Executor->>Journal: write results
    end
    
    Executor-->>CLI: ✓ Prepare done
    CLI-->>User: state/runs/<run_id>

    User->>CLI: report state/runs/<run_id>
    CLI->>Journal: read all journals
    Journal-->>CLI: aggregated data
    CLI-->>User: JSON summary
```

## Component Interactions

```mermaid
graph LR
    subgraph "Stage A: Planning"
        TC2[Task Config] --> MP2[MatrixPlanner]
        CM[Commits File] --> MP2
        MP2 --> PL[Plan JSON]
    end

    subgraph "Stage B: Execution"
        PL --> PE2[PrepareExecutor]
        PE2 --> RM2[RepoManager]
        RM2 --> WT2[Git Worktree]
        PE2 --> AI[AI Agent]
        AI --> CODE[Modified Code]
        CODE --> VAL2[Validation]
        VAL2 --> JN[Journal]
    end

    subgraph "Stage C: Reporting"
        JN --> AGG[Aggregator]
        AGG --> MET[Metrics]
        MET --> RPT[Report]
    end
```

## Data Flow Architecture

```mermaid
graph TB
    subgraph "Configuration"
        TASK[Task YAML<br/>• repo URL<br/>• commits<br/>• targets<br/>• constraints]
        BENCH[Bench YAML<br/>• container<br/>• agent config<br/>• metrics]
        CREDS[.env<br/>• LLM_MODEL<br/>• LLM_API_KEY<br/>• GITHUB_TOKEN]
    end

    subgraph "Processing Pipeline"
        P1[1. Planning]
        P2[2. Git Setup]
        P3[3. Agent Execution]
        P4[4. Validation]
        P5[5. Journaling]
    end

    subgraph "Outputs"
        PLAN2[plan.json<br/>Commit pairs]
        WORK[Worktrees<br/>Code at pre-commit]
        AGENT[Agent Branch<br/>Optimized code]
        JOURNAL[Journal Files<br/>• prompt.json<br/>• task.txt<br/>• stdout/stderr<br/>• diff_targets.json]
        REPORT[JSON Report<br/>Success metrics]
    end

    TASK --> P1
    P1 --> PLAN2
    PLAN2 --> P2
    BENCH --> P2
    P2 --> WORK
    WORK --> P3
    CREDS --> P3
    P3 --> AGENT
    AGENT --> P4
    P4 --> P5
    P5 --> JOURNAL
    JOURNAL --> REPORT
```

## File System Layout

```
perf-agents-bench/
│
├── .env                    # Runtime credentials
├── bench.yaml              # Global configuration
│
├── tasks/                  # Task definitions
│   ├── vllm.yaml
│   └── example.yaml
│
├── .work/                  # Temporary workspace
│   ├── repos/              # Base git clones
│   │   └── <repo_name>/
│   └── worktrees/          # Git worktrees
│       └── <repo>_<hash>/
│
├── state/                  # Execution state
│   ├── plan.json           # Current plan
│   └── runs/               # Run history
│       └── <run_id>/
│           └── <item_id>/
│               ├── task.txt
│               ├── prompt.json
│               ├── journal.json
│               ├── diff_targets.json
│               ├── openhands_stdout.txt
│               └── openhands_stderr.txt
│
└── bench/                  # Source code
    ├── cli.py              # CLI interface
    ├── planner.py          # Commit resolution
    ├── prepare.py          # Agent orchestration
    ├── pipeline.py         # Full pipeline
    ├── repo_manager.py     # Git operations
    ├── journal.py          # Result logging
    ├── agents/
    │   ├── base.py
    │   └── openhands.py
    ├── metrics/
    │   └── builtin/
    └── container/
        └── runtime.py
```

## Security & Isolation Layers

```mermaid
graph TB
    subgraph "Unsafe Zone 🔴"
        HOST[Host Filesystem]
        AGENT_HOST[OpenHands<br/>Direct Access]
    end

    subgraph "Semi-Safe Zone 🟡"
        WORKTREE[Git Worktree<br/>Isolated Copy]
        AGENT_WT[OpenHands<br/>Worktree Only]
    end

    subgraph "Safe Zone 🟢"
        CONTAINER[Docker Container]
        AGENT_DOCKER[OpenHands<br/>Containerized]
        VOL[Volume Mounts<br/>Read-Only]
    end

    HOST -.->|Current Default| AGENT_HOST
    WORKTREE -->|Better| AGENT_WT
    CONTAINER -->|Best Practice| AGENT_DOCKER
    VOL --> AGENT_DOCKER

    style HOST fill:#ffcccc
    style AGENT_HOST fill:#ffcccc
    style WORKTREE fill:#ffffcc
    style AGENT_WT fill:#ffffcc
    style CONTAINER fill:#ccffcc
    style AGENT_DOCKER fill:#ccffcc
```

## Key Design Decisions

1. **Git Worktrees**: Isolates each optimization attempt in separate directory
2. **Journal Pattern**: Captures all outputs for reproducibility
3. **Plugin Metrics**: Extensible metric system via registry
4. **Resume Capability**: Can restart interrupted runs
5. **Parallel Execution**: ThreadPoolExecutor for concurrent agent runs

## Current Issues

- **No sandboxing by default**: Agents run with full host access
- **Inconsistent interfaces**: Two OpenHands implementations
- **No rollback mechanism**: Failed runs can't be easily reverted
- **Limited observability**: Minimal real-time progress tracking