"""Configuration for IFBench using pydantic-settings."""

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class BenchmarkSettings(BaseSettings):
    """Settings for running IFBench benchmarks."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # API Configuration
    api_base: str = Field(
        default="http://localhost:8000/v1",
        description="Base URL for the OpenAI-compatible API",
    )
    api_key: str | None = Field(
        default=None,
        description="API key for authentication",
    )
    model: str = Field(
        default="",
        description="Model name to use for generation",
    )

    # Generation Parameters
    temperature: float = Field(
        default=1.0,
        description="Sampling temperature",
    )
    top_p: float = Field(
        default=0.95,
        description="Nucleus sampling top_p",
    )
    max_tokens: int = Field(
        default=0,
        description="Max tokens to generate; <=0 means omit (use server default)",
    )
    enable_thinking: bool = Field(
        default=True,
        description="Pass chat_template_kwargs.enable_thinking to the endpoint",
    )
    num_repeats: int = Field(
        default=8,
        description="Consensus repeats (Nemotron uses 8 for IFBench)",
    )
    seed: int | None = Field(
        default=42,
        description="Random seed for reproducibility (None for random)",
    )

    # Benchmark Parameters
    input_file: str = Field(
        default="data/IFBench_test.jsonl",
        description="Path to IFBench test file",
    )
    output_file: str = Field(
        default="data/responses.jsonl",
        description="Output file for responses",
    )
    workers: int = Field(
        default=8,
        description="Number of parallel workers",
    )
    request_timeout: float = Field(
        default=3600.0,
        description="Per-request timeout in seconds (Nemotron uses 3600)",
    )
    max_retries: int = Field(
        default=10,
        description="Retries per request on timeout/error (Nemotron uses 10)",
    )


def get_settings() -> BenchmarkSettings:
    """Load settings from environment and .env file."""
    return BenchmarkSettings()
