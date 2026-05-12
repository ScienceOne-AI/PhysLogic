import os
import random
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI


@dataclass
class GenerationResult:
    answer: str | None
    reasoning: str | None
    api_response: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "answer": self.answer,
            "reasoning": self.reasoning,
            "api_response": self.api_response,
        }


class OpenAICompatibleClient:
    """Small concurrent wrapper around OpenAI-compatible chat completions APIs."""

    def __init__(
        self,
        model_id: str,
        api_key_env: str = "OPENAI_API_KEY",
        base_url: str | None = None,
        temperature: float = 0.6,
        timeout: int = 1800,
        max_retries: int = 3,
    ) -> None:
        load_dotenv(override=False)
        api_key = os.getenv(api_key_env)
        if not api_key:
            raise ValueError(f"Missing API key: set environment variable {api_key_env}")

        self.model_id = model_id
        self.temperature = temperature
        self.max_retries = max_retries
        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url or os.getenv("OPENAI_BASE_URL"),
            timeout=timeout,
        )

    @staticmethod
    def _safe_dump(obj: Any) -> dict[str, Any]:
        if hasattr(obj, "model_dump"):
            return obj.model_dump()
        if isinstance(obj, dict):
            return obj
        return {"value": str(obj)}

    @staticmethod
    def _extract_reasoning(message: Any) -> str | None:
        reasoning = getattr(message, "reasoning_content", None)
        if reasoning:
            return reasoning
        if hasattr(message, "model_dump"):
            dumped = message.model_dump()
            return dumped.get("reasoning_content") or dumped.get("reasoning")
        return None

    def generate(self, messages: list[dict[str, str]]) -> GenerationResult:
        last_error: Exception | None = None
        for attempt in range(self.max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model_id,
                    messages=messages,
                    temperature=self.temperature,
                )
                choice = response.choices[0] if response.choices else None
                message = choice.message if choice is not None else None
                answer = getattr(message, "content", None) if message is not None else None
                reasoning = self._extract_reasoning(message) if message is not None else None
                return GenerationResult(
                    answer=(answer or "").strip(),
                    reasoning=(reasoning or "").strip() or None,
                    api_response=self._safe_dump(response),
                )
            except Exception as exc:
                last_error = exc
                if attempt < self.max_retries - 1:
                    delay = random.uniform(1, 2) if attempt == 0 else random.uniform(8, 16)
                    time.sleep(delay)

        return GenerationResult(
            answer=None,
            reasoning=None,
            api_response={"error": str(last_error) if last_error else "unknown error"},
        )

    def batch_generate(
        self,
        messages_list: list[list[dict[str, str]]],
        concurrency: int,
    ) -> list[dict[str, Any]]:
        if concurrency < 1:
            raise ValueError("concurrency must be >= 1")

        indexed_messages = list(enumerate(messages_list))
        results: list[tuple[int, GenerationResult]] = []
        with ThreadPoolExecutor(max_workers=concurrency) as pool:
            futures = {
                pool.submit(self.generate, messages): idx
                for idx, messages in indexed_messages
            }
            for future in as_completed(futures):
                idx = futures[future]
                results.append((idx, future.result()))

        results.sort(key=lambda item: item[0])
        return [result.to_dict() for _, result in results]
