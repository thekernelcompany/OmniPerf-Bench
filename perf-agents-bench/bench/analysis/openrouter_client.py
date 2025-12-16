"""OpenRouter API client for Gemini 3 Pro with thinking mode.

Handles API calls to OpenRouter, capturing full responses including
thinking content for academic analysis.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any, Optional, List

import httpx

logger = logging.getLogger(__name__)


class OpenRouterError(Exception):
    """Raised when OpenRouter API call fails."""
    pass


@dataclass
class OpenRouterConfig:
    """Configuration for OpenRouter client."""
    api_key: str
    model: str = "google/gemini-3-pro-preview"
    base_url: str = "https://openrouter.ai/api/v1"
    timeout: float = 300.0  # 5 minutes for long analyses
    max_retries: int = 3
    retry_delay: float = 5.0
    thinking_budget_tokens: int = 10000
    cache_dir: Optional[Path] = None


class OpenRouterClient:
    """Client for OpenRouter API with Gemini 3 Pro.

    Features:
    - Thinking mode support for Gemini
    - Response caching
    - Retry logic with exponential backoff
    - Full response capture (including thinking)
    """

    def __init__(self, config: OpenRouterConfig):
        self.config = config
        self.base_url = config.base_url.rstrip("/")

        if config.cache_dir:
            config.cache_dir.mkdir(parents=True, exist_ok=True)

    def _get_headers(self) -> Dict[str, str]:
        """Get request headers."""
        return {
            "Authorization": f"Bearer {self.config.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/omniperf-bench",
            "X-Title": "OmniPerf-Bench Soft Metrics Analyzer",
        }

    def _build_payload(
        self,
        prompt: str,
        enable_thinking: bool = True,
        temperature: float = 0.0,
        max_tokens: int = 8192,
    ) -> Dict[str, Any]:
        """Build request payload."""
        payload: Dict[str, Any] = {
            "model": self.config.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        # Enable thinking for Gemini models
        if enable_thinking and "gemini" in self.config.model.lower():
            payload["provider"] = {
                "order": ["Google"],
                "allow_fallbacks": False,
            }
            # Gemini thinking config via transforms or model-specific params
            # Note: OpenRouter may use different param names
            payload["transforms"] = ["middle-out"]

        return payload

    def _get_cache_path(self, prompt: str) -> Optional[Path]:
        """Get cache file path for a prompt."""
        if not self.config.cache_dir:
            return None
        import hashlib
        prompt_hash = hashlib.sha256(prompt.encode()).hexdigest()[:16]
        return self.config.cache_dir / f"{prompt_hash}.json"

    def _load_from_cache(self, prompt: str) -> Optional[Dict[str, Any]]:
        """Load response from cache if available."""
        cache_path = self._get_cache_path(prompt)
        if cache_path and cache_path.exists():
            try:
                data = json.loads(cache_path.read_text())
                logger.info(f"Loaded response from cache: {cache_path.name}")
                return data
            except Exception as e:
                logger.warning(f"Failed to load cache: {e}")
        return None

    def _save_to_cache(self, prompt: str, response: Dict[str, Any]):
        """Save response to cache."""
        cache_path = self._get_cache_path(prompt)
        if cache_path:
            try:
                cache_path.write_text(json.dumps(response, indent=2))
                logger.info(f"Saved response to cache: {cache_path.name}")
            except Exception as e:
                logger.warning(f"Failed to save cache: {e}")

    async def analyze(
        self,
        prompt: str,
        enable_thinking: bool = True,
        use_cache: bool = True,
    ) -> Dict[str, Any]:
        """Send analysis request to OpenRouter.

        Args:
            prompt: The analysis prompt
            enable_thinking: Whether to enable thinking mode
            use_cache: Whether to use response caching

        Returns:
            Dict containing:
                - content: Main response content
                - thinking: Thinking content (if available)
                - usage: Token usage statistics
                - model: Model used
                - raw: Raw API response
        """
        # Check cache first
        if use_cache:
            cached = self._load_from_cache(prompt)
            if cached:
                return cached

        payload = self._build_payload(prompt, enable_thinking)
        endpoint = f"{self.base_url}/chat/completions"

        last_error = None
        for attempt in range(self.config.max_retries):
            try:
                start_time = time.time()

                async with httpx.AsyncClient(timeout=self.config.timeout) as client:
                    response = await client.post(
                        endpoint,
                        headers=self._get_headers(),
                        json=payload,
                    )

                duration = time.time() - start_time

                if response.status_code != 200:
                    error_text = response.text
                    logger.error(f"OpenRouter API error: {response.status_code} - {error_text}")
                    raise OpenRouterError(f"API returned {response.status_code}: {error_text}")

                data = response.json()

                # Extract response components
                result = self._parse_response(data, duration)
                result["prompt"] = prompt

                # Cache the result
                if use_cache:
                    self._save_to_cache(prompt, result)

                return result

            except httpx.TimeoutException as e:
                last_error = e
                logger.warning(f"Request timeout (attempt {attempt + 1}/{self.config.max_retries})")
            except httpx.RequestError as e:
                last_error = e
                logger.warning(f"Request error (attempt {attempt + 1}/{self.config.max_retries}): {e}")
            except OpenRouterError:
                raise
            except Exception as e:
                last_error = e
                logger.error(f"Unexpected error: {e}")

            # Exponential backoff
            if attempt < self.config.max_retries - 1:
                delay = self.config.retry_delay * (2 ** attempt)
                logger.info(f"Retrying in {delay}s...")
                await asyncio.sleep(delay)

        raise OpenRouterError(f"Failed after {self.config.max_retries} attempts: {last_error}")

    def _parse_response(self, data: Dict[str, Any], duration: float) -> Dict[str, Any]:
        """Parse API response into structured format."""
        choices = data.get("choices", [])
        if not choices:
            raise OpenRouterError("No choices in response")

        message = choices[0].get("message", {})
        content = message.get("content", "")

        # Extract thinking content if present
        # Gemini may include thinking in different ways
        thinking = ""

        # Check for thinking in message metadata
        if "thinking" in message:
            thinking = message["thinking"]

        # Check for thinking markers in content
        if "<thinking>" in content and "</thinking>" in content:
            import re
            thinking_match = re.search(r"<thinking>(.*?)</thinking>", content, re.DOTALL)
            if thinking_match:
                thinking = thinking_match.group(1).strip()
                # Remove thinking from main content
                content = re.sub(r"<thinking>.*?</thinking>", "", content, flags=re.DOTALL).strip()

        # Get usage stats
        usage = data.get("usage", {})

        return {
            "content": content,
            "thinking": thinking,
            "usage": {
                "prompt_tokens": usage.get("prompt_tokens", 0),
                "completion_tokens": usage.get("completion_tokens", 0),
                "total_tokens": usage.get("total_tokens", 0),
            },
            "model": data.get("model", self.config.model),
            "duration_s": duration,
            "raw": data,
        }

    def extract_json(self, response: Dict[str, Any]) -> Dict[str, Any]:
        """Extract JSON from response content.

        Handles markdown code blocks and raw JSON.
        """
        content = response.get("content", "")

        # Try to find JSON in code blocks
        import re
        json_match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", content)
        if json_match:
            try:
                return json.loads(json_match.group(1))
            except json.JSONDecodeError:
                pass

        # Try parsing the whole content as JSON
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            pass

        # Try to find JSON object in content
        json_match = re.search(r"\{[\s\S]*\}", content)
        if json_match:
            try:
                return json.loads(json_match.group(0))
            except json.JSONDecodeError:
                pass

        logger.warning("Could not extract JSON from response")
        return {}


def create_client(
    api_key: Optional[str] = None,
    model: str = "google/gemini-3-pro-preview",
    cache_dir: Optional[Path] = None,
) -> OpenRouterClient:
    """Create OpenRouter client with sensible defaults.

    Args:
        api_key: OpenRouter API key (defaults to OPENROUTER_API_KEY env var)
        model: Model to use
        cache_dir: Directory for response caching

    Returns:
        Configured OpenRouterClient
    """
    key = api_key or os.getenv("OPENROUTER_API_KEY")
    if not key:
        raise ValueError("OpenRouter API key required (set OPENROUTER_API_KEY or pass api_key)")

    config = OpenRouterConfig(
        api_key=key,
        model=model,
        cache_dir=cache_dir,
    )

    return OpenRouterClient(config)
