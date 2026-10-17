import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class LLMConfig:
    provider: str
    api_key: str
    base_url: str
    model: str
    max_output_tokens: int


def load_llm_config() -> LLMConfig:
    provider = os.getenv("LLM_PROVIDER", "euri").lower().strip()

    if provider not in {"euri", "mesh"}:
        raise ValueError(
            "LLM_PROVIDER must be either 'euri' or 'mesh'."
        )

    prefix = provider.upper()

    api_key = os.getenv(f"{prefix}_API_KEY", "").strip()
    base_url = os.getenv(f"{prefix}_BASE_URL", "").strip()
    model = os.getenv(f"{prefix}_MODEL", "").strip()

    if not api_key:
        raise ValueError(f"{prefix}_API_KEY is missing.")

    if not base_url:
        raise ValueError(f"{prefix}_BASE_URL is missing.")

    if not model:
        raise ValueError(f"{prefix}_MODEL is missing.")

    max_tokens = int(
        os.getenv("LLM_MAX_OUTPUT_TOKENS", "500")
    )

    if max_tokens < 1:
        raise ValueError("LLM_MAX_OUTPUT_TOKENS must be positive.")

    return LLMConfig(
        provider=provider,
        api_key=api_key,
        base_url=base_url,
        model=model,
        max_output_tokens=max_tokens,
    )