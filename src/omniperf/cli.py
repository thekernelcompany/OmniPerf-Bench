"""Command-line interface for OmniPerf-Bench."""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

try:
    import yaml
except ImportError:
    yaml = None

from omniperf.pipeline import Pipeline, PipelineConfig


def setup_logging(verbose: bool = False):
    """Configure logging."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler("omniperf.log", mode="w"),
        ],
    )


def load_config(config_path: Path) -> PipelineConfig:
    """Load configuration from YAML file."""
    if yaml is None:
        raise ImportError("PyYAML is required. Install with: pip install pyyaml")

    with open(config_path) as f:
        data = yaml.safe_load(f)

    return PipelineConfig(
        input_source=data.get("input_source") or data.get("extractions_dir", ""),
        repo_path=Path(data.get("repo_path", ".")),
        llm_provider=data.get("llm_provider", "openai"),
        llm_model=data.get("llm_model", "gpt-4o-mini"),
        llm_temperature=data.get("llm_temperature", 0.1),
        llm_max_tokens=data.get("llm_max_tokens", 4096),
        num_test_samples=data.get("num_test_samples", 1),
        executor=data.get("executor", "local"),
        timeout_seconds=data.get("timeout_seconds", 300),
        num_runs=data.get("num_runs", 3),
        output_dir=Path(data.get("output_dir", "data")),
        dataset_name=data.get("dataset_name", "omniperf_dataset"),
        min_speedup=data.get("min_speedup", 1.0),
        push_to_hf=data.get("push_to_hf", False),
        hf_repo=data.get("hf_repo"),
    )


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="OmniPerf-Bench: Performance optimization benchmark framework"
    )
    parser.add_argument(
        "config",
        type=Path,
        help="Path to YAML configuration file",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate config without running pipeline",
    )

    args = parser.parse_args()

    setup_logging(args.verbose)
    logger = logging.getLogger(__name__)

    # Load config
    if not args.config.exists():
        logger.error(f"Config file not found: {args.config}")
        sys.exit(1)

    try:
        config = load_config(args.config)
    except Exception as e:
        logger.error(f"Failed to load config: {e}")
        sys.exit(1)

    logger.info(f"Loaded config from {args.config}")

    if args.dry_run:
        logger.info("Dry run - config is valid")
        print(f"Input source: {config.input_source}")
        print(f"Repository: {config.repo_path}")
        print(f"LLM: {config.llm_provider}/{config.llm_model}")
        print(f"Executor: {config.executor}")
        print(f"Output: {config.output_dir}/{config.dataset_name}.jsonl")
        sys.exit(0)

    # Run pipeline
    try:
        pipeline = Pipeline(config)
        output_path = pipeline.run()
        print(f"\nDataset written to: {output_path}")
    except Exception as e:
        logger.exception(f"Pipeline failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
