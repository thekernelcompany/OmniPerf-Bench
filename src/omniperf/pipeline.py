"""Main pipeline orchestrating the full dataset generation flow."""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from omniperf.data import CommitExtraction
from omniperf.input import create_loader
from omniperf.generate import TestGenerator, GeneratorConfig
from omniperf.execute import LocalExecutor, ExecutorConfig
from omniperf.build import DatasetBuilder, BuilderConfig

logger = logging.getLogger(__name__)


@dataclass
class PipelineConfig:
    """Full pipeline configuration."""
    # Input
    input_source: str  # Path or hf:// URL

    # Repository
    repo_path: Path

    # Generation
    llm_provider: str = "openai"
    llm_model: str = "gpt-4o-mini"
    llm_temperature: float = 0.1
    llm_max_tokens: int = 4096
    num_test_samples: int = 1

    # Execution
    executor: str = "local"  # local, modal, skypilot
    timeout_seconds: int = 300
    num_runs: int = 3

    # Output
    output_dir: Path = field(default_factory=lambda: Path("data"))
    dataset_name: str = "omniperf_dataset"
    min_speedup: float = 1.0
    push_to_hf: bool = False
    hf_repo: Optional[str] = None

    def __post_init__(self):
        self.repo_path = Path(self.repo_path)
        self.output_dir = Path(self.output_dir)


class Pipeline:
    """Main pipeline for dataset generation."""

    def __init__(self, config: PipelineConfig):
        self.config = config

        # Initialize components
        self.loader = create_loader(config.input_source)

        self.generator = TestGenerator(GeneratorConfig(
            provider=config.llm_provider,
            model=config.llm_model,
            temperature=config.llm_temperature,
            max_tokens=config.llm_max_tokens,
            num_samples=config.num_test_samples,
        ))

        self.executor = self._create_executor()

        self.builder = DatasetBuilder(BuilderConfig(
            output_dir=config.output_dir,
            dataset_name=config.dataset_name,
            min_speedup=config.min_speedup,
            push_to_hf=config.push_to_hf,
            hf_repo=config.hf_repo,
        ))

    def _create_executor(self):
        """Create executor based on config."""
        exec_config = ExecutorConfig(
            repo_path=self.config.repo_path,
            timeout_seconds=self.config.timeout_seconds,
            num_runs=self.config.num_runs,
        )

        if self.config.executor == "local":
            return LocalExecutor(exec_config)
        elif self.config.executor == "modal":
            # Placeholder for Modal executor
            raise NotImplementedError("Modal executor not yet implemented. Use 'local' for now.")
        elif self.config.executor == "skypilot":
            # Placeholder for SkyPilot executor
            raise NotImplementedError("SkyPilot executor not yet implemented. Use 'local' for now.")
        else:
            raise ValueError(f"Unknown executor: {self.config.executor}")

    def run(self) -> Path:
        """Run the full pipeline and return path to output dataset."""
        logger.info("Starting OmniPerf-Bench pipeline")

        # Load extractions
        extractions = self.loader.load()
        logger.info(f"Loaded {len(extractions)} commit extractions")

        # Process each extraction
        for i, extraction in enumerate(extractions):
            logger.info(f"Processing {i+1}/{len(extractions)}: {extraction.commit_hash[:8]}")

            try:
                # Generate tests
                tests = self.generator.generate(extraction)
                if not tests:
                    logger.warning(f"No tests generated for {extraction.commit_hash}")
                    continue
                logger.info(f"Generated {len(tests)} tests")

                # Execute tests
                result = self.executor.execute(extraction, tests)
                logger.info(f"Executed tests on {extraction.commit_hash[:8]}")

                # Add to dataset
                record = self.builder.add_result(extraction, tests, result)
                if record:
                    logger.info(f"Added record: {record.instance_id} (speedup: {record.human_performance:.2f}x)")

            except Exception as e:
                logger.error(f"Failed to process {extraction.commit_hash}: {e}")
                continue

        # Build final dataset
        output_path = self.builder.build()
        logger.info(f"Pipeline complete. Output: {output_path}")

        return output_path
