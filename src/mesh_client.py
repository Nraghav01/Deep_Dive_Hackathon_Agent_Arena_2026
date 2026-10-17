from openai import OpenAI

from src.config import load_llm_config


class LLMClient:
    def __init__(self):
        self.config = load_llm_config()

        self.client = OpenAI(
            api_key=self.config.api_key,
            base_url=self.config.base_url,
            timeout=30.0,
            max_retries=1,
        )

    def generate(
        self,
        prompt: str,
        system_prompt: str = (
            "You are a careful research assistant. "
            "Follow the instructions and do not invent evidence."
        ),
        max_output_tokens: int | None = None,
        temperature: float = 0,
    ) -> str:
        # Enforce a configured ceiling, even if a caller
        # accidentally requests more tokens.
        requested_limit = (
            max_output_tokens
            if max_output_tokens is not None
            else self.config.max_output_tokens
        )

        if requested_limit < 1:
            raise ValueError("Token limit must be positive.")

        token_limit = min(
            requested_limit,
            self.config.max_output_tokens,
        )

        response = self.client.chat.completions.create(
            model=self.config.model,
            messages=[
                {
                    "role": "system",
                    "content": system_prompt,
                },
                {
                    "role": "user",
                    "content": prompt,
                },
            ],
            max_tokens=token_limit,
            temperature=temperature,
        )

        # Track usage without logging prompts or credentials.
        usage = response.usage
        if usage:
            print(
                f"[LLM] provider={self.config.provider} "
                f"model={self.config.model} "
                f"prompt_tokens={usage.prompt_tokens} "
                f"completion_tokens={usage.completion_tokens} "
                f"total_tokens={usage.total_tokens}"
            )

        content = response.choices[0].message.content

        if not content:
            raise RuntimeError("The LLM returned an empty response.")

        return content.strip()