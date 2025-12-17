"""Main soft metrics analyzer for agent benchmark runs.

Orchestrates the analysis pipeline:
1. Load all run artifacts
2. Extract quantitative metrics directly
3. Build prompt and call LLM for qualitative analysis
4. Parse LLM response into structured metrics
5. Return complete RunAnalysis
"""

from __future__ import annotations

import asyncio
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional, Callable

from .schemas import (
    RunMetadata,
    QuantitativeMetrics,
    ToolDistribution,
    PatchMetrics,
    QualitativeScores,
    QualityScore,
    CategoricalMetrics,
    FreeFormAnalysis,
    LLMAnalysisResult,
    RunAnalysis,
    ApproachCategory,
    FailureCategory,
    ToolUsagePattern,
    # V3 Academic-grade metrics
    TrajectoryMetrics,
    InteractionMetrics,
    PatchSimilarityMetrics,
    LLMRawScores,
    # V4 Patch Quality Analysis (categories + discussion, no scores)
    PatchQualityAnalysis,
    BottleneckTargetAnalysis,
    BottleneckTargetCategory,
    OptimizationTechniqueAnalysis,
    OptimizationTechnique,
    ApproachComparisonAnalysis,
    PatchApproachCategory,
    SpeedupLikelihoodAnalysis,
    SpeedupLikelihood,
    FailureModeAnalysis,
    PatchFailureMode,
    PatchObservations,
)
from .openrouter_client import OpenRouterClient, OpenRouterConfig, create_client
from .run_loader import (
    RunLoader,
    extract_patch_metrics,
    extract_error_metrics,
    count_tool_calls,
    TrajectoryParser,
)
from .prompts import build_analysis_prompt, build_patch_quality_prompt
from .patch_comparator import (
    PatchComparator,
    load_human_patch_from_dataset,
    find_dataset_for_repo,
)

logger = logging.getLogger(__name__)


class SoftMetricsAnalyzer:
    """Analyze agent benchmark runs and extract soft metrics.

    Uses Gemini 3 Pro via OpenRouter to extract qualitative metrics
    from agent trajectories and outputs.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "google/gemini-3-pro-preview",
        cache_dir: Optional[Path] = None,
    ):
        """Initialize the analyzer.

        Args:
            api_key: OpenRouter API key (optional for discover_runs only)
            model: Model to use for analysis
            cache_dir: Directory for caching LLM responses
        """
        self._api_key = api_key
        self._cache_dir = cache_dir
        self._client: Optional[OpenRouterClient] = None
        self.model = model

    @property
    def client(self) -> OpenRouterClient:
        """Get the OpenRouter client, creating it lazily if needed."""
        if self._client is None:
            if not self._api_key:
                raise ValueError(
                    "OpenRouter API key required for LLM analysis. "
                    "Set OPENROUTER_API_KEY or pass api_key."
                )
            self._client = create_client(
                api_key=self._api_key,
                model=self.model,
                cache_dir=self._cache_dir,
            )
        return self._client

    def discover_runs(
        self,
        state_root: Path,
        repo_filter: Optional[str] = None,
        agent_filter: Optional[str] = None,
        model_filter: Optional[str] = None,
    ) -> List[Path]:
        """Discover all run directories matching filters.

        Args:
            state_root: Root state directory (contains runs/)
            repo_filter: Filter by repo name (vllm, sglang)
            agent_filter: Filter by agent type (trae, codex, openhands)
            model_filter: Filter by model name (claude-sonnet-45, gpt-5)

        Returns:
            List of item directories containing journal.json
        """
        runs_dir = state_root / "runs"
        if not runs_dir.exists():
            logger.warning(f"Runs directory not found: {runs_dir}")
            return []

        item_dirs: List[Path] = []

        # Walk the hierarchical structure: runs/{repo}/{agent}/{model}/{timestamp}/{item}
        for repo_dir in runs_dir.iterdir():
            if not repo_dir.is_dir():
                continue
            if repo_filter and repo_dir.name != repo_filter:
                continue

            for agent_dir in repo_dir.iterdir():
                if not agent_dir.is_dir():
                    continue
                if agent_filter and agent_dir.name != agent_filter:
                    continue

                for model_dir in agent_dir.iterdir():
                    if not model_dir.is_dir():
                        continue
                    if model_filter and model_filter not in model_dir.name:
                        continue

                    for timestamp_dir in model_dir.iterdir():
                        if not timestamp_dir.is_dir():
                            continue

                        for item_dir in timestamp_dir.iterdir():
                            if not item_dir.is_dir():
                                continue
                            if (item_dir / "journal.json").exists():
                                item_dirs.append(item_dir)

        logger.info(f"Discovered {len(item_dirs)} runs")
        return sorted(item_dirs)

    def _extract_quantitative_metrics(self, data: Dict[str, Any]) -> QuantitativeMetrics:
        """Extract quantitative metrics directly from loaded data."""
        journal = data.get("journal", {})
        run_summary = data.get("run_summary", {})
        diff_targets = data.get("diff_targets", {})
        metadata = data.get("metadata", {})

        # Get agent-specific info
        agent_type = metadata.get("agent", "unknown")
        agent_info = journal.get(agent_type, {})

        # Get metrics from various sources
        metrics_data = journal.get("metrics", {})
        agent_metrics = run_summary.get("agent", {})
        patch_stats = agent_metrics.get("patch_stats", {})

        # Extract patch metrics from actual diff
        patch_data = extract_patch_metrics(data.get("patch", ""))

        # Extract error metrics from stderr
        error_data = extract_error_metrics(data.get("stderr", ""))

        # Count tool calls from stdout
        tool_dist = count_tool_calls(data.get("stdout", ""), agent_type)

        # Build tool distribution
        tools = ToolDistribution(
            bash=tool_dist.get("bash", 0),
            editor=tool_dist.get("editor", 0),
            read=tool_dist.get("read", 0),
            search=tool_dist.get("grep", 0) + tool_dist.get("glob", 0),
            other=sum(v for k, v in tool_dist.items() if k not in ["bash", "editor", "read", "grep", "glob"]),
        )

        # Build patch metrics
        patch = PatchMetrics(
            generated=patch_data["lines_added"] > 0 or patch_data["lines_removed"] > 0,
            lines_added=patch_data["lines_added"],
            lines_removed=patch_data["lines_removed"],
            hunks=patch_data["hunks"],
            files_changed=diff_targets.get("changed", []),
            files_allowed=diff_targets.get("allowed", []),
            files_disallowed=diff_targets.get("disallowed", []),
            compliance_ok=diff_targets.get("ok", True),
        )

        return QuantitativeMetrics(
            duration_s=agent_info.get("duration_s", 0) or 0,
            time_to_first_edit_s=metrics_data.get("time_to_first_edit_s", 0) or 0,
            step_count=len(data.get("trajectory", {}).get("agent_steps", [])),
            tool_calls=tools,
            input_tokens=self._sum_tokens(data.get("trajectory", {}), "input"),
            output_tokens=self._sum_tokens(data.get("trajectory", {}), "output"),
            patch=patch,
            error_count=error_data["error_count"],
            warning_count=error_data["warning_count"],
            exception_types=error_data["exception_types"],
            status=journal.get("status", "unknown"),
            returncode=agent_info.get("returncode", 0),
        )

    def _sum_tokens(self, trajectory: Dict[str, Any], token_type: str) -> int:
        """Sum tokens from trajectory LLM interactions."""
        total = 0
        for interaction in trajectory.get("llm_interactions", []):
            usage = interaction.get("response", {}).get("usage", {})
            if token_type == "input":
                total += usage.get("input_tokens", 0) or usage.get("prompt_tokens", 0)
            else:
                total += usage.get("output_tokens", 0) or usage.get("completion_tokens", 0)
        return total

    def _parse_trajectory_metrics(self, trajectory_data: Dict[str, Any]) -> Optional[TrajectoryMetrics]:
        """Parse trajectory.json into TrajectoryMetrics.

        Args:
            trajectory_data: Loaded trajectory.json content

        Returns:
            TrajectoryMetrics or None if parsing fails
        """
        if not trajectory_data:
            return None

        try:
            parser = TrajectoryParser(trajectory_data)
            metrics = parser.parse_all()

            # Convert interaction dicts to InteractionMetrics objects
            interactions = [
                InteractionMetrics(**interaction_data)
                for interaction_data in metrics.pop("interactions", [])
            ]

            return TrajectoryMetrics(
                **metrics,
                interactions=interactions,
            )
        except Exception as e:
            logger.warning(f"Failed to parse trajectory metrics: {e}")
            return None

    def _compare_patches(
        self,
        agent_patch: str,
        repo: str,
        commits: Dict[str, str],
        data_dir: Optional[Path] = None,
    ) -> Optional[PatchSimilarityMetrics]:
        """Compare agent patch against human reference.

        Args:
            agent_patch: Agent's generated patch
            repo: Repository name
            commits: Dict with 'pre', 'human' commit hashes from run metadata.
                     The 'human' key contains the commit hash of the human's
                     solution, which maps to 'commit_hash' in the dataset.
            data_dir: Data directory for finding datasets

        Returns:
            PatchSimilarityMetrics or None if comparison not possible
        """
        if not agent_patch:
            return None

        # Get human commit hash from run metadata (maps to commit_hash in dataset)
        human_commit = commits.get("human", "")
        if not human_commit:
            return PatchSimilarityMetrics(
                human_patch_available=False,
                agent_files=[],
                human_files=[],
            )

        # Find dataset for this repo
        if data_dir is None:
            # Try common data directories relative to state root
            data_dir = Path.cwd() / "data"

        dataset_path = find_dataset_for_repo(repo, data_dir)

        human_patch = ""
        source = ""
        if dataset_path:
            human_patch, source = load_human_patch_from_dataset(
                dataset_path, human_commit
            )

        if not human_patch:
            return PatchSimilarityMetrics(
                human_patch_available=False,
                human_patch_source=f"Not found for commit {human_commit[:7]}",
                agent_files=self._extract_patch_files(agent_patch),
            )

        # Compare patches
        try:
            comparator = PatchComparator(agent_patch, human_patch)
            metrics = comparator.compare()
            metrics["human_patch_source"] = source
            return PatchSimilarityMetrics(**metrics)
        except Exception as e:
            logger.warning(f"Failed to compare patches: {e}")
            return None

    def _extract_patch_files(self, patch_content: str) -> List[str]:
        """Extract file paths from a patch."""
        files = set()
        for line in patch_content.split("\n"):
            if line.startswith("+++ b/"):
                files.add(line[6:].split("\t")[0])
            elif line.startswith("+++ "):
                parts = line[4:].split("\t")[0].split(" ")[0]
                if parts != "/dev/null" and not parts.startswith("b/"):
                    files.add(parts)
                elif parts.startswith("b/"):
                    files.add(parts[2:])
        return sorted(files)

    def _extract_raw_scores(self, response: Dict[str, Any]) -> Optional[LLMRawScores]:
        """Extract raw LLM scores from response.

        Args:
            response: Full LLM response dict

        Returns:
            LLMRawScores or None
        """
        json_data = self.client.extract_json(response)

        if not json_data:
            return None

        return LLMRawScores(
            quality_scores=json_data.get("quality_scores", {}),
            categorical=json_data.get("categorical", {}),
            quantitative_assessment=json_data.get("quantitative_assessment", {}),
            detailed_analysis=json_data.get("detailed_analysis", {}),
            recommendations=json_data.get("recommendations", {}),
        )

    async def _analyze_patch_quality(
        self,
        human_patch: str,
        agent_patch: str,
        task_description: str,
        rule_based_metrics: Optional[Dict[str, Any]] = None,
    ) -> Optional[PatchQualityAnalysis]:
        """Analyze patch quality using LLM categorical assessment.

        Compares agent's patch to human reference using categories + discussion,
        inspired by GSO Benchmark (arxiv:2505.23671v3).

        Args:
            human_patch: Human reference patch (unified diff)
            agent_patch: Agent's generated patch (unified diff)
            task_description: Brief description of the optimization task
            rule_based_metrics: Pre-computed metrics from PatchComparator

        Returns:
            PatchQualityAnalysis or None if analysis fails
        """
        if not human_patch:
            return PatchQualityAnalysis(
                human_patch_available=False,
                analysis_model=self.model,
            )

        # Build minimal prompt
        prompt = build_patch_quality_prompt(
            task_description=task_description,
            human_patch=human_patch,
            agent_patch=agent_patch,
            rule_based_metrics=rule_based_metrics,
        )

        try:
            # Call LLM
            logger.info(f"Analyzing patch quality with {self.model}...")
            response = await self.client.analyze(prompt)

            # Extract JSON from response
            json_data = self.client.extract_json(response)

            if not json_data:
                logger.warning("Could not extract JSON from patch quality LLM response")
                return PatchQualityAnalysis(
                    human_patch_available=True,
                    analysis_model=self.model,
                )

            # Parse bottleneck target
            bt_data = json_data.get("bottleneck_target", {})
            bottleneck_target = None
            if bt_data:
                try:
                    category = BottleneckTargetCategory(bt_data.get("category", "other"))
                except ValueError:
                    category = BottleneckTargetCategory.OTHER
                bottleneck_target = BottleneckTargetAnalysis(
                    category=category,
                    human_target=bt_data.get("human_target", ""),
                    agent_target=bt_data.get("agent_target", ""),
                    discussion=bt_data.get("discussion", ""),
                )

            # Parse optimization techniques
            ot_data = json_data.get("optimization_techniques", {})
            optimization_techniques = None
            if ot_data:
                def parse_techniques(techniques: List[str]) -> List[OptimizationTechnique]:
                    result = []
                    for t in techniques:
                        try:
                            result.append(OptimizationTechnique(t))
                        except ValueError:
                            result.append(OptimizationTechnique.OTHER)
                    return result

                optimization_techniques = OptimizationTechniqueAnalysis(
                    human_techniques=parse_techniques(ot_data.get("human_techniques", [])),
                    agent_techniques=parse_techniques(ot_data.get("agent_techniques", [])),
                    technique_overlap=ot_data.get("technique_overlap", False),
                    discussion=ot_data.get("discussion", ""),
                )

            # Parse approach comparison
            ac_data = json_data.get("approach_comparison", {})
            approach_comparison = None
            if ac_data:
                try:
                    category = PatchApproachCategory(ac_data.get("category", "other"))
                except ValueError:
                    category = PatchApproachCategory.OTHER
                approach_comparison = ApproachComparisonAnalysis(
                    category=category,
                    discussion=ac_data.get("discussion", ""),
                )

            # Parse speedup likelihood
            sl_data = json_data.get("speedup_likelihood", {})
            speedup_likelihood = None
            if sl_data:
                try:
                    category = SpeedupLikelihood(sl_data.get("category", "other"))
                except ValueError:
                    category = SpeedupLikelihood.OTHER
                speedup_likelihood = SpeedupLikelihoodAnalysis(
                    category=category,
                    discussion=sl_data.get("discussion", ""),
                )

            # Parse failure mode
            fm_data = json_data.get("failure_mode", {})
            failure_mode = None
            if fm_data:
                try:
                    category = PatchFailureMode(fm_data.get("category", "other"))
                except ValueError:
                    category = PatchFailureMode.OTHER
                failure_mode = FailureModeAnalysis(
                    category=category,
                    discussion=fm_data.get("discussion", ""),
                )

            # Parse observations
            obs_data = json_data.get("observations", {})
            observations = None
            if obs_data:
                observations = PatchObservations(
                    key_differences=obs_data.get("key_differences", []),
                    agent_strengths=obs_data.get("agent_strengths", []),
                    agent_weaknesses=obs_data.get("agent_weaknesses", []),
                    benchmark_needed=obs_data.get("benchmark_needed", ""),
                )

            # Get token usage
            usage = response.get("usage", {})
            total_tokens = usage.get("total_tokens", 0)

            return PatchQualityAnalysis(
                human_patch_available=True,
                analysis_model=self.model,
                analysis_tokens=total_tokens,
                bottleneck_target=bottleneck_target,
                optimization_techniques=optimization_techniques,
                approach_comparison=approach_comparison,
                speedup_likelihood=speedup_likelihood,
                failure_mode=failure_mode,
                observations=observations,
            )

        except Exception as e:
            logger.error(f"Patch quality analysis failed: {e}")
            return PatchQualityAnalysis(
                human_patch_available=True,
                analysis_model=self.model,
            )

    def _parse_llm_response(self, response: Dict[str, Any]) -> tuple[
        QualitativeScores, CategoricalMetrics, FreeFormAnalysis
    ]:
        """Parse LLM response into structured metrics."""
        # Extract JSON from response
        json_data = self.client.extract_json(response)

        if not json_data:
            logger.warning("Could not extract JSON from LLM response")
            return QualitativeScores(), CategoricalMetrics(), FreeFormAnalysis()

        # Parse quality scores
        scores_data = json_data.get("quality_scores", {})

        def build_quality_score(key: str) -> QualityScore:
            data = scores_data.get(key, {})
            if isinstance(data, dict):
                # Extract sub-scores
                sub_scores = {k: float(v) for k, v in data.items()
                              if k not in ["score", "justification"] and isinstance(v, (int, float))}
                return QualityScore(
                    score=float(data.get("score", 0)),
                    sub_scores=sub_scores,
                    justification=data.get("justification", ""),
                )
            return QualityScore(score=float(data) if data else 0.0)

        qualitative = QualitativeScores(
            code_understanding=build_quality_score("code_understanding"),
            task_alignment=build_quality_score("task_alignment"),
            approach_quality=build_quality_score("approach_quality"),
            execution_quality=build_quality_score("execution_quality"),
        )

        # Parse categorical metrics
        cat_data = json_data.get("categorical", {})

        # Map string to enum safely
        approach = cat_data.get("approach_category", "unknown")
        try:
            approach_enum = ApproachCategory(approach)
        except ValueError:
            approach_enum = ApproachCategory.UNKNOWN

        tool_pattern = cat_data.get("tool_usage_pattern", "unknown")
        try:
            tool_enum = ToolUsagePattern(tool_pattern)
        except ValueError:
            tool_enum = ToolUsagePattern.UNKNOWN

        failure = cat_data.get("failure_category", "none")
        try:
            failure_enum = FailureCategory(failure)
        except ValueError:
            failure_enum = FailureCategory.NONE

        categorical = CategoricalMetrics(
            approach_category=approach_enum,
            tool_usage_pattern=tool_enum,
            failure_category=failure_enum,
        )

        # Parse free-form analysis
        analysis_data = json_data.get("detailed_analysis", {})
        recommendations_data = json_data.get("recommendations", {})

        free_form = FreeFormAnalysis(
            summary=analysis_data.get("summary", ""),
            key_decisions=analysis_data.get("key_decisions", []),
            optimization_techniques=analysis_data.get("optimization_techniques", []),
            missed_opportunities=analysis_data.get("missed_opportunities", []),
            error_recovery_strategy=analysis_data.get("error_recovery_strategy", ""),
            strengths=analysis_data.get("strengths", []),
            weaknesses=analysis_data.get("weaknesses", []),
            recommendations=(
                recommendations_data.get("prompting_improvements", []) +
                recommendations_data.get("capability_additions", [])
            ),
        )

        return qualitative, categorical, free_form

    async def analyze_run(
        self,
        item_dir: Path,
        skip_llm: bool = False,
    ) -> RunAnalysis:
        """Analyze a single run directory.

        Args:
            item_dir: Path to the run item directory
            skip_llm: If True, only extract quantitative metrics (no LLM call)

        Returns:
            Complete RunAnalysis object
        """
        start_time = time.time()

        # Load all data
        loader = RunLoader(item_dir)
        data = loader.load_all()

        # Build metadata
        meta_data = data.get("metadata", {})
        meta = RunMetadata(
            repo=meta_data.get("repo", "unknown"),
            agent=meta_data.get("agent", "unknown"),
            model=meta_data.get("model", "unknown"),
            model_full=meta_data.get("model_full", meta_data.get("model", "unknown")),
            timestamp=meta_data.get("timestamp", "unknown"),
            item_id=meta_data.get("item_id", item_dir.name),
            task_id=meta_data.get("task_id", ""),
            commits=meta_data.get("commits", {}),
            source_files=data.get("source_files", {}),
        )

        # Extract quantitative metrics
        quant = self._extract_quantitative_metrics(data)

        # Initialize qualitative metrics
        qualitative = QualitativeScores()
        categorical = CategoricalMetrics()
        free_form = FreeFormAnalysis()
        llm_result = LLMAnalysisResult(model=self.model)

        # V3: Parse trajectory metrics
        trajectory_metrics = self._parse_trajectory_metrics(data.get("trajectory", {}))

        # V3: Compare patches against human reference
        data_dir = item_dir.parent.parent.parent.parent.parent / "data"
        patch_similarity = self._compare_patches(
            agent_patch=data.get("patch", ""),
            repo=meta.repo,
            commits=meta.commits,
            data_dir=data_dir,
        )

        # Load human patch for V4 patch quality analysis
        human_patch = ""
        human_commit = meta.commits.get("human", "")
        if human_commit:
            dataset_path = find_dataset_for_repo(meta.repo, data_dir)
            if dataset_path:
                human_patch, _ = load_human_patch_from_dataset(dataset_path, human_commit)

        # Initialize raw LLM scores and patch quality
        llm_raw: Optional[LLMRawScores] = None
        patch_quality: Optional[PatchQualityAnalysis] = None

        # Call LLM for qualitative analysis if not skipped
        if not skip_llm:
            try:
                # Build prompt
                prompt = build_analysis_prompt(data)
                llm_result.prompt = prompt

                # Call LLM
                logger.info(f"Analyzing {item_dir.name} with {self.model}...")
                response = await self.client.analyze(prompt)

                # Parse response
                qualitative, categorical, free_form = self._parse_llm_response(response)

                # V3: Extract raw LLM scores
                llm_raw = self._extract_raw_scores(response)

                # Store LLM result details
                llm_result.thinking_content = response.get("thinking", "")
                llm_result.response_content = response.get("content", "")
                llm_result.prompt_tokens = response.get("usage", {}).get("prompt_tokens", 0)
                llm_result.completion_tokens = response.get("usage", {}).get("completion_tokens", 0)
                llm_result.total_tokens = response.get("usage", {}).get("total_tokens", 0)
                llm_result.duration_s = response.get("duration_s", 0)
                llm_result.timestamp = datetime.now().isoformat()

            except Exception as e:
                logger.error(f"LLM analysis failed for {item_dir}: {e}")

            # V4: Patch quality analysis (separate LLM call with minimal context)
            if human_patch:
                try:
                    # Get task description from prompt data
                    prompt_data = data.get("prompt", {})
                    task_description = data.get("task", prompt_data.get("description", ""))

                    # Get rule-based metrics for context
                    rule_based_metrics = None
                    if patch_similarity:
                        rule_based_metrics = patch_similarity.model_dump()

                    patch_quality = await self._analyze_patch_quality(
                        human_patch=human_patch,
                        agent_patch=data.get("patch", ""),
                        task_description=task_description,
                        rule_based_metrics=rule_based_metrics,
                    )
                except Exception as e:
                    logger.error(f"Patch quality analysis failed for {item_dir}: {e}")

        # Build complete analysis with V3 and V4 fields
        analysis = RunAnalysis(
            meta=meta,
            quantitative=quant,
            qualitative=qualitative,
            categorical=categorical,
            analysis=free_form,
            llm=llm_result,
            # V3 academic-grade metrics
            trajectory=trajectory_metrics,
            patch_similarity=patch_similarity,
            llm_raw=llm_raw,
            # V4 patch quality analysis (categories + discussion)
            patch_quality=patch_quality,
            analysis_duration_s=time.time() - start_time,
        )

        return analysis

    async def analyze_batch(
        self,
        item_dirs: List[Path],
        progress_callback: Optional[Callable[[int, int, str], None]] = None,
        skip_llm: bool = False,
        max_concurrent: int = 3,
    ) -> List[RunAnalysis]:
        """Analyze multiple runs with progress tracking.

        Args:
            item_dirs: List of run directories to analyze
            progress_callback: Callback(current, total, item_id) for progress
            skip_llm: If True, skip LLM analysis
            max_concurrent: Maximum concurrent analyses

        Returns:
            List of RunAnalysis objects
        """
        results: List[RunAnalysis] = []
        semaphore = asyncio.Semaphore(max_concurrent)

        async def analyze_with_semaphore(item_dir: Path, index: int) -> Optional[RunAnalysis]:
            async with semaphore:
                try:
                    result = await self.analyze_run(item_dir, skip_llm=skip_llm)
                    if progress_callback:
                        progress_callback(index + 1, len(item_dirs), item_dir.name)
                    return result
                except Exception as e:
                    logger.error(f"Failed to analyze {item_dir}: {e}")
                    if progress_callback:
                        progress_callback(index + 1, len(item_dirs), f"{item_dir.name} (FAILED)")
                    return None

        # Create tasks
        tasks = [
            analyze_with_semaphore(item_dir, i)
            for i, item_dir in enumerate(item_dirs)
        ]

        # Run all tasks
        completed = await asyncio.gather(*tasks, return_exceptions=True)

        # Filter results
        for result in completed:
            if isinstance(result, RunAnalysis):
                results.append(result)
            elif isinstance(result, Exception):
                logger.error(f"Analysis task failed: {result}")

        return results


async def analyze_directory(
    state_root: Path,
    api_key: str,
    repo: Optional[str] = None,
    agent: Optional[str] = None,
    model_filter: Optional[str] = None,
    skip_llm: bool = False,
    cache_dir: Optional[Path] = None,
) -> List[RunAnalysis]:
    """Convenience function to analyze all runs in a directory.

    Args:
        state_root: Root state directory
        api_key: OpenRouter API key
        repo: Filter by repo
        agent: Filter by agent
        model_filter: Filter by model
        skip_llm: Skip LLM analysis
        cache_dir: Cache directory for LLM responses

    Returns:
        List of RunAnalysis objects
    """
    analyzer = SoftMetricsAnalyzer(
        api_key=api_key,
        cache_dir=cache_dir,
    )

    item_dirs = analyzer.discover_runs(
        state_root,
        repo_filter=repo,
        agent_filter=agent,
        model_filter=model_filter,
    )

    if not item_dirs:
        logger.warning("No runs found")
        return []

    def progress(current: int, total: int, item: str):
        print(f"\rAnalyzing... {current}/{total} - {item}", end="", flush=True)

    results = await analyzer.analyze_batch(
        item_dirs,
        progress_callback=progress,
        skip_llm=skip_llm,
    )

    print()  # Newline after progress
    return results
