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

        # Initialize tracking counters
        self.calls_made = 0
        self.total_tokens_used = 0

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

        # Budget guardrails
        if self.calls_made >= self.config.max_calls_per_run:
            raise RuntimeError("Workflow LLM call budget exhausted.")

        # Approximate prompt tokens using 4 characters per token.
        # This is a safety estimate, not an exact tokenizer.
        estimated_input_tokens = (
            len(system_prompt) + len(prompt) + 3
        ) // 4

        if (
            self.total_tokens_used
            + estimated_input_tokens
            + token_limit
            > self.config.max_total_tokens_per_run
        ):
            raise RuntimeError("Workflow token budget would be exceeded.")

        # Increment call counter immediately before executing the request
        self.calls_made += 1

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
            max_tokens=token_limit, #some implementations might require max_completion_tokens instead of max_tokens
            temperature=temperature,
        )

        # Track usage without logging prompts or credentials.
        usage = response.usage
        content = response.choices[0].message.content
        if usage:
            print(
                f"[LLM] provider={self.config.provider} "
                f"model={self.config.model} "
                f"prompt_tokens={usage.prompt_tokens} "
                f"completion_tokens={usage.completion_tokens} "
                f"total_tokens={usage.total_tokens}"
            )
            self.total_tokens_used += usage.total_tokens
    
        else:
            # Conservative fallback if the LLM provider does not return usage.
            self.total_tokens_used += (
                estimated_input_tokens + len(content or "") // 4
            )

        if not content:
            raise RuntimeError("The LLM returned an empty response.")

        return content.strip()

