"""Performance test generation using LLMs."""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import List, Optional

from omniperf.data import CommitExtraction, PerformanceTest

logger = logging.getLogger(__name__)


@dataclass
class GeneratorConfig:
    """Configuration for test generation."""
    provider: str = "openai"  # openai, anthropic, bedrock
    model: str = "gpt-4o-mini"
    temperature: float = 0.1
    max_tokens: int = 4096
    num_samples: int = 1


class TestGenerator:
    """Generate performance tests from commit extractions using LLMs."""

    SYSTEM_PROMPT = """You are an expert at writing performance tests for Python code.
Given a commit that optimizes code, write a performance test that:
1. Exercises the optimized code path
2. Prints timing in format: "Execution time: X.XXXXs"
3. Uses realistic inputs that trigger the optimization
4. Is self-contained and runnable

Output ONLY the Python code, no explanations."""

    USER_PROMPT_TEMPLATE = """Write a performance test for this optimization commit:

Repository: {repo_name}
Commit message: {message}

Affected APIs: {apis}

Diff:
```
{diff}
```

Write a complete, runnable Python test that measures the performance of the optimized code.
The test MUST print timing as: print(f"Execution time: {{duration:.4f}}s")
"""

    def __init__(self, config: GeneratorConfig):
        self.config = config
        self._client = None

    def _get_client(self):
        """Lazy-load the appropriate LLM client."""
        if self._client is not None:
            return self._client

        if self.config.provider == "openai":
            from openai import OpenAI
            self._client = OpenAI()
        elif self.config.provider == "anthropic":
            from anthropic import Anthropic
            self._client = Anthropic()
        elif self.config.provider == "bedrock":
            import boto3
            self._client = boto3.client("bedrock-runtime")
        else:
            raise ValueError(f"Unknown provider: {self.config.provider}")

        return self._client

    def generate(self, extraction: CommitExtraction) -> List[PerformanceTest]:
        """Generate performance tests for a commit extraction."""
        prompt = self.USER_PROMPT_TEMPLATE.format(
            repo_name=extraction.repo_name or "unknown",
            message=extraction.message,
            apis=", ".join(extraction.affected_apis) or "N/A",
            diff=extraction.diff_text[:8000],  # Truncate long diffs
        )

        tests = []
        for i in range(self.config.num_samples):
            try:
                code = self._call_llm(prompt)
                code = self._extract_code(code)

                test = PerformanceTest(
                    test_id=f"{extraction.commit_hash[:8]}_{i}",
                    code=code,
                    target_api=extraction.affected_apis[0] if extraction.affected_apis else None,
                    generation_model=f"{self.config.provider}/{self.config.model}",
                )
                tests.append(test)

            except Exception as e:
                logger.warning(f"Failed to generate test {i}: {e}")

        return tests

    def _call_llm(self, prompt: str) -> str:
        """Call the LLM and return response text."""
        client = self._get_client()

        if self.config.provider == "openai":
            response = client.chat.completions.create(
                model=self.config.model,
                messages=[
                    {"role": "system", "content": self.SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
            )
            return response.choices[0].message.content

        elif self.config.provider == "anthropic":
            response = client.messages.create(
                model=self.config.model,
                max_tokens=self.config.max_tokens,
                system=self.SYSTEM_PROMPT,
                messages=[{"role": "user", "content": prompt}],
            )
            return response.content[0].text

        elif self.config.provider == "bedrock":
            import json
            body = json.dumps({
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": self.config.max_tokens,
                "system": self.SYSTEM_PROMPT,
                "messages": [{"role": "user", "content": prompt}],
            })
            response = client.invoke_model(
                modelId=self.config.model,
                body=body,
            )
            result = json.loads(response["body"].read())
            return result["content"][0]["text"]

        raise ValueError(f"Unknown provider: {self.config.provider}")

    def _extract_code(self, response: str) -> str:
        """Extract Python code from LLM response."""
        # Try to find code blocks
        if "```python" in response:
            start = response.find("```python") + 9
            end = response.find("```", start)
            if end > start:
                return response[start:end].strip()

        if "```" in response:
            start = response.find("```") + 3
            end = response.find("```", start)
            if end > start:
                return response[start:end].strip()

        # Return as-is if no code blocks
        return response.strip()
