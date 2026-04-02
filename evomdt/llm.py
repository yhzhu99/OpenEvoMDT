from __future__ import annotations

import json
from typing import Any, Callable, Protocol, TypeVar

from openai import AsyncOpenAI
from pydantic import BaseModel, Field
from tenacity import retry, stop_after_attempt, wait_exponential

from .config import LLMConfig

T = TypeVar("T")


class ChatMessage(BaseModel):
    role: str
    content: str


class LLMTrace(BaseModel):
    raw_content: str
    parsed: Any
    model_name: str
    usage: dict[str, Any] = Field(default_factory=dict)


class StructuredLLM(Protocol):
    async def generate_structured(
        self,
        messages: list[ChatMessage],
        parser: Callable[[str], T],
    ) -> LLMTrace:
        ...


def parse_json_content(content: str) -> Any:
    stripped = content.strip()
    if not stripped:
        raise ValueError("Response content was empty.")
    decoder = json.JSONDecoder()
    candidates = [stripped]
    for candidate in candidates:
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            pass
        for index, char in enumerate(candidate):
            if char not in "{[":
                continue
            try:
                parsed, _ = decoder.raw_decode(candidate[index:])
                return parsed
            except json.JSONDecodeError:
                continue
    raise ValueError("Response did not contain valid JSON.")


class OpenAIChatProvider:
    def __init__(self, config: LLMConfig, client: AsyncOpenAI | None = None):
        if not config.api_key:
            raise ValueError(
                f"Missing API key for model '{config.model_name}'. "
                f"Set {config.api_key_env or 'the configured api_key'}."
            )
        self.config = config
        self.client = client or AsyncOpenAI(api_key=config.api_key, base_url=config.base_url)

    @retry(wait=wait_exponential(multiplier=1, min=1, max=8), stop=stop_after_attempt(3))
    async def generate_structured(
        self,
        messages: list[ChatMessage],
        parser: Callable[[str], T],
    ) -> LLMTrace:
        completion = await self.client.chat.completions.create(
            model=self.config.model_name,
            messages=[message.model_dump() for message in messages],
            response_format={"type": "json_object"},
        )
        raw_content = completion.choices[0].message.content or ""
        parsed = parser(raw_content)
        usage = {}
        if getattr(completion, "usage", None) is not None:
            usage = {
                "prompt_tokens": getattr(completion.usage, "prompt_tokens", None),
                "completion_tokens": getattr(completion.usage, "completion_tokens", None),
                "total_tokens": getattr(completion.usage, "total_tokens", None),
            }
        return LLMTrace(
            raw_content=raw_content,
            parsed=parsed,
            model_name=self.config.model_name,
            usage=usage,
        )
