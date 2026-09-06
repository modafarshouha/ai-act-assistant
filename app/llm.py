"""LLM interface, a deterministic double, and an Ollama client.

The double is the default so a fresh clone runs with no model download. It does
not generate anything; it quotes the passages it was handed. That keeps the
graph tests deterministic, and it stops fluent output being mistaken for
evidence that the pipeline worked.
"""

import re
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import httpx

from app.config import Settings, get_settings
from app.prompts import TASK_DECOMPOSE, TASK_SYNTHESIZE, task_of


class LLMError(RuntimeError):
    pass


@dataclass(frozen=True)
class LLMResponse:
    text: str
    provider: str
    model: str
    latency_s: float = 0.0


@runtime_checkable
class LLMProvider(Protocol):
    @property
    def name(self) -> str: ...

    @property
    def model(self) -> str: ...

    def complete(self, prompt: str, system: str | None = None) -> LLMResponse: ...

    def health(self) -> bool: ...


# "and" followed by a question word, which is where two questions got stuck
# together.
_SPLIT = re.compile(
    r",?\s+and\s+(?=(?:when|what|which|who|how|whether|does|do|is|are|can)\b)", re.IGNORECASE
)
_PASSAGE = re.compile(r"^\[(\d+)\]\s*(.+?)(?=^\[\d+\]|\Z)", re.MULTILINE | re.DOTALL)


class DummyLLM:
    """Stand-in provider. Splits on conjunctions, answers by quoting."""

    @property
    def name(self) -> str:
        return "dummy"

    @property
    def model(self) -> str:
        return "deterministic-test-double"

    def complete(self, prompt: str, system: str | None = None) -> LLMResponse:
        task = task_of(prompt)
        if task == TASK_DECOMPOSE:
            text = self._decompose(prompt)
        elif task == TASK_SYNTHESIZE:
            text = self._synthesize(prompt)
        else:
            text = "This provider does not generate text."
        return LLMResponse(text=text, provider=self.name, model=self.model)

    def health(self) -> bool:
        return True

    @staticmethod
    def _decompose(prompt: str) -> str:
        match = re.search(r"^Question:\s*(.+)$", prompt, re.MULTILINE)
        if not match:
            return ""
        question = match.group(1).strip()
        parts = [part.strip(" ,;") for part in _SPLIT.split(question) if part.strip(" ,;")]
        if len(parts) < 2:
            return question
        restored = []
        for part in parts:
            part = part[0].upper() + part[1:] if part else part
            restored.append(part if part.endswith("?") else f"{part}?")
        return "\n".join(restored)

    def _synthesize(self, prompt: str) -> str:
        block = prompt.split("Passages from the regulations:", 1)
        passages = _PASSAGE.findall(block[1].split("Question:", 1)[0]) if len(block) > 1 else []

        lines = []
        for number, body in passages[:3]:
            text = " ".join(body.split())
            # Drop the citation header line and quote the opening of the body.
            if "\n" in body:
                text = " ".join(body.split("\n", 1)[1].split())
            lines.append(f"{self._opening(text)} [{number}]")

        if not lines:
            return "The retrieved passages do not contain an answer to this question."

        answer = " ".join(lines)
        computed = self._computed(prompt)
        return f"{computed} {answer}" if computed else answer

    @staticmethod
    def _opening(text: str, limit: int = 320) -> str:
        if len(text) <= limit:
            return text
        cut = text.rfind(" ", 0, limit)
        return text[: cut if cut > 60 else limit].rstrip(" ,;") + "..."

    @staticmethod
    def _computed(prompt: str) -> str:
        match = re.search(r"do not recalculate:\n(.+?)\n\nPassages", prompt, re.DOTALL)
        return " ".join(match.group(1).split()) if match else ""


class OllamaLLM:
    def __init__(self, settings: Settings | None = None):
        self.settings = settings or get_settings()
        self._client = httpx.Client(
            base_url=self.settings.ollama_base_url, timeout=self.settings.ollama_timeout_s
        )

    @property
    def name(self) -> str:
        return "ollama"

    @property
    def model(self) -> str:
        return self.settings.ollama_model

    def complete(self, prompt: str, system: str | None = None) -> LLMResponse:
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "keep_alive": -1,
            "options": {"temperature": 0, "num_predict": 320, "num_thread": 4},
        }
        if system:
            payload["system"] = system

        try:
            response = self._client.post("/api/generate", json=payload)
        except httpx.TimeoutException as exc:
            raise LLMError(
                f"{self.model} did not answer within {self.settings.ollama_timeout_s:.0f}s; "
                f"a larger model on CPU may simply be too slow"
            ) from exc
        except httpx.HTTPError as exc:
            raise LLMError(
                f"cannot reach Ollama at {self.settings.ollama_base_url}: {exc}"
            ) from exc

        if response.status_code != 200:
            raise LLMError(
                f"Ollama returned {response.status_code}; is the model pulled? "
                f"(ollama pull {self.model})"
            )

        text = response.json().get("response", "").strip()
        if not text:
            raise LLMError(f"{self.model} returned an empty completion")
        return LLMResponse(text=text, provider=self.name, model=self.model)

    def health(self) -> bool:
        try:
            tags = self._client.get("/api/tags", timeout=5.0).json()
        except (httpx.HTTPError, ValueError):
            return False
        names = {entry.get("name", "") for entry in tags.get("models", [])}
        return any(name == self.model or name.startswith(f"{self.model}:") for name in names)


def get_llm(settings: Settings | None = None) -> LLMProvider:
    settings = settings or get_settings()
    return OllamaLLM(settings) if settings.llm_provider == "ollama" else DummyLLM()
