"""Output writer for hierarchical analysis results.

Writes analysis results to a hierarchical directory structure:
    state/analysis/{repo}/{agent}/{model}/{timestamp}/{item_id}/
        - analysis.json          # Complete Pydantic-validated analysis
        - llm_prompt.txt         # Exact prompt sent to LLM
        - llm_response.json      # Full LLM response with thinking traces
        - llm_analysis.json      # V3: Raw JSON returned by LLM (structured)
        - llm_raw_scores.json    # V3: Extracted scores and categorical data only
        - metrics_summary.json   # Quick-reference compact summary
        - trajectory_metrics.json # V3: Detailed per-step metrics
        - patch_similarity.json   # V3: Agent vs human patch comparison
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional

from .schemas import RunAnalysis

logger = logging.getLogger(__name__)


class OutputWriter:
    """Write analysis results to hierarchical directory structure."""

    def __init__(self, output_root: Path):
        """Initialize the output writer.

        Args:
            output_root: Root directory for analysis output
                        (e.g., state/analysis/)
        """
        self.root = Path(output_root)
        self.root.mkdir(parents=True, exist_ok=True)

    def _get_output_dir(self, analysis: RunAnalysis) -> Path:
        """Get the output directory for an analysis.

        Structure: {root}/{repo}/{agent}/{model}/{timestamp}/{item_id}/
        """
        return (
            self.root /
            analysis.meta.repo /
            analysis.meta.agent /
            analysis.meta.model /
            analysis.meta.timestamp /
            analysis.meta.item_id
        )

    def write(self, analysis: RunAnalysis) -> Path:
        """Write a single analysis to the output directory.

        Args:
            analysis: The RunAnalysis to write

        Returns:
            Path to the output directory
        """
        out_dir = self._get_output_dir(analysis)
        out_dir.mkdir(parents=True, exist_ok=True)

        # Write complete analysis
        analysis_path = out_dir / "analysis.json"
        analysis_path.write_text(analysis.to_json())
        logger.debug(f"Wrote analysis to {analysis_path}")

        # Write LLM prompt
        if analysis.llm.prompt:
            prompt_path = out_dir / "llm_prompt.txt"
            prompt_path.write_text(analysis.llm.prompt)
            logger.debug(f"Wrote prompt to {prompt_path}")

        # Write LLM response (including thinking)
        if analysis.llm.response_content or analysis.llm.thinking_content:
            response_path = out_dir / "llm_response.json"
            response_data = {
                "model": analysis.llm.model,
                "provider": analysis.llm.provider,
                "thinking_enabled": analysis.llm.thinking_enabled,
                "timestamp": analysis.llm.timestamp,
                "duration_s": analysis.llm.duration_s,
                "usage": {
                    "prompt_tokens": analysis.llm.prompt_tokens,
                    "completion_tokens": analysis.llm.completion_tokens,
                    "total_tokens": analysis.llm.total_tokens,
                },
                "thinking": analysis.llm.thinking_content,
                "response": analysis.llm.response_content,
            }
            response_path.write_text(json.dumps(response_data, indent=2))
            logger.debug(f"Wrote LLM response to {response_path}")

        # Write metrics summary (quick reference)
        summary_path = out_dir / "metrics_summary.json"
        summary_path.write_text(json.dumps(analysis.metrics_summary(), indent=2))
        logger.debug(f"Wrote metrics summary to {summary_path}")

        # V3: Write LLM analysis (full structured JSON from LLM)
        if analysis.llm_raw:
            llm_analysis_path = out_dir / "llm_analysis.json"
            llm_analysis_data = {
                "quality_scores": analysis.llm_raw.quality_scores,
                "categorical": analysis.llm_raw.categorical,
                "quantitative_assessment": analysis.llm_raw.quantitative_assessment,
                "detailed_analysis": analysis.llm_raw.detailed_analysis,
                "recommendations": analysis.llm_raw.recommendations,
            }
            llm_analysis_path.write_text(json.dumps(llm_analysis_data, indent=2))
            logger.debug(f"Wrote LLM analysis to {llm_analysis_path}")

            # V3: Write raw scores (just scores and categories for quick reference)
            raw_scores_path = out_dir / "llm_raw_scores.json"
            raw_scores_data = {
                "scores": {
                    "code_understanding": analysis.qualitative.code_understanding.score,
                    "task_alignment": analysis.qualitative.task_alignment.score,
                    "approach_quality": analysis.qualitative.approach_quality.score,
                    "execution_quality": analysis.qualitative.execution_quality.score,
                    "overall": analysis.qualitative.overall_score,
                },
                "sub_scores": {
                    "code_understanding": analysis.qualitative.code_understanding.sub_scores,
                    "task_alignment": analysis.qualitative.task_alignment.sub_scores,
                    "approach_quality": analysis.qualitative.approach_quality.sub_scores,
                    "execution_quality": analysis.qualitative.execution_quality.sub_scores,
                },
                "categorical": {
                    "approach_category": analysis.categorical.approach_category.value,
                    "tool_usage_pattern": analysis.categorical.tool_usage_pattern.value,
                    "failure_category": analysis.categorical.failure_category.value,
                },
            }
            raw_scores_path.write_text(json.dumps(raw_scores_data, indent=2))
            logger.debug(f"Wrote raw scores to {raw_scores_path}")

        # V3: Write trajectory metrics
        if analysis.trajectory:
            trajectory_path = out_dir / "trajectory_metrics.json"
            trajectory_path.write_text(json.dumps(
                analysis.trajectory.model_dump(mode="json"),
                indent=2
            ))
            logger.debug(f"Wrote trajectory metrics to {trajectory_path}")

        # V3: Write patch similarity
        if analysis.patch_similarity:
            similarity_path = out_dir / "patch_similarity.json"
            similarity_path.write_text(json.dumps(
                analysis.patch_similarity.model_dump(mode="json"),
                indent=2
            ))
            logger.debug(f"Wrote patch similarity to {similarity_path}")

        logger.info(f"Wrote analysis for {analysis.meta.item_id} to {out_dir}")
        return out_dir

    def write_batch(self, analyses: List[RunAnalysis]) -> List[Path]:
        """Write multiple analyses.

        Args:
            analyses: List of RunAnalysis objects

        Returns:
            List of output directory paths
        """
        paths = []
        for analysis in analyses:
            try:
                path = self.write(analysis)
                paths.append(path)
            except Exception as e:
                logger.error(f"Failed to write analysis for {analysis.meta.item_id}: {e}")
        return paths

    def write_aggregate_report(
        self,
        analyses: List[RunAnalysis],
        filename: str = "aggregate_report.json",
    ) -> Path:
        """Write an aggregate report summarizing all analyses.

        Args:
            analyses: List of RunAnalysis objects
            filename: Name of the aggregate report file

        Returns:
            Path to the aggregate report
        """
        # Group by various dimensions
        by_repo: Dict[str, List[Dict]] = {}
        by_agent: Dict[str, List[Dict]] = {}
        by_model: Dict[str, List[Dict]] = {}
        by_agent_model: Dict[str, List[Dict]] = {}

        all_metrics = []

        for analysis in analyses:
            summary = analysis.metrics_summary()
            all_metrics.append(summary)

            # Group by repo
            repo = analysis.meta.repo
            if repo not in by_repo:
                by_repo[repo] = []
            by_repo[repo].append(summary)

            # Group by agent
            agent = analysis.meta.agent
            if agent not in by_agent:
                by_agent[agent] = []
            by_agent[agent].append(summary)

            # Group by model
            model = analysis.meta.model
            if model not in by_model:
                by_model[model] = []
            by_model[model].append(summary)

            # Group by agent+model
            agent_model = f"{agent}/{model}"
            if agent_model not in by_agent_model:
                by_agent_model[agent_model] = []
            by_agent_model[agent_model].append(summary)

        # Calculate aggregate statistics
        def calc_stats(metrics: List[Dict]) -> Dict[str, Any]:
            if not metrics:
                return {}

            n = len(metrics)
            success_count = sum(1 for m in metrics if m["execution"]["status"] == "success")

            # Aggregate execution metrics
            avg_duration = sum(m["execution"]["duration_s"] for m in metrics) / n
            avg_steps = sum(m["execution"]["steps"] for m in metrics) / n
            avg_tools = sum(m["execution"]["tool_calls"] for m in metrics) / n
            avg_errors = sum(m["execution"]["errors"] for m in metrics) / n
            patch_rate = sum(1 for m in metrics if m["patch"]["generated"]) / n

            # Aggregate quality scores
            avg_understanding = sum(m["scores"]["code_understanding"] for m in metrics) / n
            avg_alignment = sum(m["scores"]["task_alignment"] for m in metrics) / n
            avg_approach = sum(m["scores"]["approach_quality"] for m in metrics) / n
            avg_execution = sum(m["scores"]["execution_quality"] for m in metrics) / n
            avg_overall = sum(m["scores"]["overall"] for m in metrics) / n

            # Count approach categories
            approach_dist = {}
            failure_dist = {}
            for m in metrics:
                approach = m["classification"]["approach"]
                approach_dist[approach] = approach_dist.get(approach, 0) + 1

                failure = m["classification"]["failure_reason"]
                if failure and failure != "none":
                    failure_dist[failure] = failure_dist.get(failure, 0) + 1

            return {
                "count": n,
                "success_count": success_count,
                "success_rate": success_count / n if n > 0 else 0,
                "patch_rate": patch_rate,
                "execution": {
                    "avg_duration_s": round(avg_duration, 2),
                    "avg_steps": round(avg_steps, 2),
                    "avg_tool_calls": round(avg_tools, 2),
                    "avg_errors": round(avg_errors, 2),
                },
                "scores": {
                    "avg_code_understanding": round(avg_understanding, 2),
                    "avg_task_alignment": round(avg_alignment, 2),
                    "avg_approach_quality": round(avg_approach, 2),
                    "avg_execution_quality": round(avg_execution, 2),
                    "avg_overall": round(avg_overall, 2),
                },
                "distributions": {
                    "approach_categories": approach_dist,
                    "failure_categories": failure_dist,
                },
            }

        # Build aggregate report
        report = {
            "generated_at": datetime.now().isoformat(),
            "total_runs": len(analyses),
            "overall": calc_stats(all_metrics),
            "by_repo": {repo: calc_stats(metrics) for repo, metrics in by_repo.items()},
            "by_agent": {agent: calc_stats(metrics) for agent, metrics in by_agent.items()},
            "by_model": {model: calc_stats(metrics) for model, metrics in by_model.items()},
            "by_agent_model": {key: calc_stats(metrics) for key, metrics in by_agent_model.items()},
        }

        # Write report
        report_path = self.root / filename
        report_path.write_text(json.dumps(report, indent=2))
        logger.info(f"Wrote aggregate report to {report_path}")

        return report_path

    def list_analyses(self) -> List[Path]:
        """List all analysis.json files in the output directory.

        Returns:
            List of paths to analysis.json files
        """
        return sorted(self.root.rglob("analysis.json"))

    def load_analysis(self, analysis_path: Path) -> Optional[RunAnalysis]:
        """Load a RunAnalysis from a file.

        Args:
            analysis_path: Path to analysis.json

        Returns:
            RunAnalysis object or None if loading fails
        """
        try:
            data = json.loads(analysis_path.read_text())
            return RunAnalysis.model_validate(data)
        except Exception as e:
            logger.error(f"Failed to load analysis from {analysis_path}: {e}")
            return None

    def load_all_analyses(self) -> List[RunAnalysis]:
        """Load all analyses from the output directory.

        Returns:
            List of RunAnalysis objects
        """
        analyses = []
        for path in self.list_analyses():
            analysis = self.load_analysis(path)
            if analysis:
                analyses.append(analysis)
        return analyses
