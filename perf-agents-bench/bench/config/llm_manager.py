"""
LLM Configuration Management for OpenHands Benchmarking

This module manages LLM provider configurations for OpenHands, supporting
multiple providers and enabling easy benchmarking across different models.
"""

import os
from pathlib import Path
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


class LLMConfigError(Exception):
    """Raised when LLM configuration is invalid or incomplete."""
    pass


class LLMProvider:
    """Base class for LLM provider configurations."""

    def __init__(self, name: str, model: str, api_key: Optional[str] = None):
        self.name = name
        self.model = model
        self.api_key = api_key or os.getenv(f"{name.upper()}_API_KEY")

    def get_config_section(self) -> Dict[str, Any]:
        """Return the LLM configuration section for this provider."""
        raise NotImplementedError

    def validate(self) -> bool:
        """Validate that this provider is properly configured."""
        return self.api_key is not None


class AnthropicProvider(LLMProvider):
    """Anthropic Claude configuration."""

    def __init__(self, model: str = "claude-sonnet-4-20250514", api_key: Optional[str] = None):
        super().__init__("anthropic", model, api_key)

    def get_config_section(self) -> Dict[str, Any]:
        return {
            "model": self.model,
            "api_key": self.api_key,
            "base_url": "https://api.anthropic.com",
            "custom_llm_provider": "anthropic",
            "max_input_tokens": 0,
            "max_output_tokens": 0,
            "input_cost_per_token": 0.0,
            "output_cost_per_token": 0.0,
            "ollama_base_url": "",
            "drop_params": False
        }


class OpenAIProvider(LLMProvider):
    """OpenAI GPT configuration."""

    def __init__(self, model: str = "gpt-5-2025-08-07", api_key: Optional[str] = None):
        super().__init__("openai", model, api_key)

    def get_config_section(self) -> Dict[str, Any]:
        return {
            "model": self.model,
            "api_key": self.api_key,
            "base_url": "https://api.openai.com/v1",
            "custom_llm_provider": "openai",
            "max_input_tokens": 128000,
            "max_output_tokens": 4096,
            "input_cost_per_token": 0.0,
            "output_cost_per_token": 0.0,
            "ollama_base_url": "",
            "drop_params": False
        }


class LLMConfigManager:
    """
    Manages LLM configurations for OpenHands benchmarking.

    This class handles:
    - Multiple LLM provider configurations
    - TOML file generation for OpenHands
    - Configuration validation
    - Cost tracking setup
    """

    def __init__(self, config_dir: Optional[Path] = None):
        self.config_dir = config_dir or Path("config")
        # Ensure config_dir is always an absolute path
        self.config_dir = self.config_dir.resolve()
        self.config_dir.mkdir(parents=True, exist_ok=True)
        self.providers = {}
        self._setup_providers()

    def _setup_providers(self):
        """Set up available LLM providers."""
        self.providers["anthropic"] = AnthropicProvider()
        self.providers["openai"] = OpenAIProvider()

    def add_provider(self, provider: LLMProvider):
        """Add a custom LLM provider."""
        self.providers[provider.name] = provider

    def get_provider(self, name: str) -> Optional[LLMProvider]:
        """Get a configured provider by name."""
        return self.providers.get(name)

    def list_available_providers(self) -> list[str]:
        """List all configured providers."""
        return list(self.providers.keys())

    def validate_all_providers(self) -> Dict[str, bool]:
        """Validate all configured providers."""
        results = {}
        for name, provider in self.providers.items():
            results[name] = provider.validate()
            if not results[name]:
                logger.warning(f"Provider {name} is not properly configured")
        return results

    def create_llm_config(self, provider_name: str) -> Path:
        """
        Create a TOML configuration file for the specified LLM provider.

        Args:
            provider_name: Name of the LLM provider

        Returns:
            Path to the created configuration file

        Raises:
            LLMConfigError: If provider is not configured or invalid
        """
        if provider_name not in self.providers:
            raise LLMConfigError(f"Unknown provider: {provider_name}")

        provider = self.providers[provider_name]

        if not provider.validate():
            raise LLMConfigError(f"Provider {provider_name} is not properly configured")

        config_data = provider.get_config_section()

        # Generate TOML content
        toml_content = "[llm]\n"
        for key, value in config_data.items():
            if isinstance(value, str):
                toml_content += f'{key} = "{value}"\n'
            elif isinstance(value, bool):
                toml_content += f'{key} = {str(value).lower()}\n'
            else:
                toml_content += f'{key} = {value}\n'

        # Write configuration file
        config_file = self.config_dir / f"llm_{provider_name}.toml"
        config_file.write_text(toml_content)

        logger.info(f"Created LLM config file: {config_file}")
        return config_file

    def create_main_config(self, provider_name: str) -> Path:
        """
        Create a main OpenHands configuration file that includes LLM config inline.

        Args:
            provider_name: Name of the LLM provider

        Returns:
            Path to the created main configuration file
        """
        if provider_name not in self.providers:
            raise LLMConfigError(f"Unknown provider: {provider_name}")

        provider = self.providers[provider_name]
        if not provider.validate():
            raise LLMConfigError(f"Provider {provider_name} is not properly configured")

        config_data = provider.get_config_section()

        # Create main config with LLM section inline
        config_content = "[llm]\n"
        for key, value in config_data.items():
            if isinstance(value, str):
                config_content += f'{key} = "{value}"\n'
            elif isinstance(value, bool):
                config_content += f'{key} = {str(value).lower()}\n'
            else:
                config_content += f'{key} = {value}\n'

        # Add minimal agent section
        config_content += "\n[agent]\n"
        config_content += "# Agent configuration\n"

        # Add core section with local runtime (no Docker required)
        config_content += "\n[core]\n"
        config_content += "runtime = \"local\"\n"
        config_content += "# Local runtime - no Docker required\n"
        config_content += "# Workspace mapping is handled via SANDBOX_VOLUMES environment variable\n"

        config_file = self.config_dir / f"main_{provider_name}.toml"
        config_file.write_text(config_content)

        logger.info(f"Created main config file: {config_file}")
        return config_file

    def setup_for_benchmarking(self, provider_name: str) -> tuple[Path, Path]:
        """
        Set up configuration for benchmarking.

        Args:
            provider_name: Name of the LLM provider to use

        Returns:
            Tuple of (main_config_path, None) - second value is None since we use single config

        Raises:
            LLMConfigError: If setup fails
        """
        try:
            # Create main configuration with LLM config inline
            main_config = self.create_main_config(provider_name)

            logger.info(f"Benchmarking setup complete for {provider_name}")
            # Return main config as both values for compatibility
            return main_config, main_config

        except Exception as e:
            raise LLMConfigError(f"Failed to setup benchmarking for {provider_name}: {e}")

    def get_cost_estimate(self, provider_name: str, estimated_tokens: int) -> float:
        """
        Get cost estimate for a provider based on token usage.

        Args:
            provider_name: Name of the LLM provider
            estimated_tokens: Estimated number of tokens to be used

        Returns:
            Estimated cost in USD
        """
        # This is a simplified cost estimation
        # In practice, you'd want to track actual usage
        provider = self.providers.get(provider_name)
        if not provider:
            return 0.0

        # Simplified pricing (update with actual rates)
        pricing = {
            "anthropic": {"input": 3.0, "output": 15.0},  # per million tokens
            "openai": {"input": 10.0, "output": 30.0}     # per million tokens
        }

        rates = pricing.get(provider_name, {"input": 0.0, "output": 0.0})

        # Assume 80% input, 20% output for estimation
        input_cost = (estimated_tokens * 0.8) * rates["input"] / 1_000_000
        output_cost = (estimated_tokens * 0.2) * rates["output"] / 1_000_000

        return input_cost + output_cost


def create_benchmarking_config(
    providers: list[str],
    config_dir: Optional[Path] = None
) -> Dict[str, Any]:
    """
    Create a complete benchmarking configuration.

    Args:
        providers: List of provider names to configure
        config_dir: Directory to store configuration files

    Returns:
        Configuration dictionary with paths and settings
    """
    manager = LLMConfigManager(config_dir)

    config = {
        "providers": {},
        "benchmarking": {
            "max_parallel_runs": 2,
            "cost_tracking": True,
            "result_correlation": True,
            "comparative_analysis": True
        }
    }

    for provider_name in providers:
        try:
            llm_config, agent_config = manager.setup_for_benchmarking(provider_name)
            config["providers"][provider_name] = {
                "llm_config": str(llm_config),
                "agent_config": str(agent_config),
                "available": True
            }
        except LLMConfigError as e:
            logger.warning(f"Failed to configure {provider_name}: {e}")
            config["providers"][provider_name] = {
                "available": False,
                "error": str(e)
            }

    return config
