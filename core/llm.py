"""Shared LLM factory: Ollama (local) or Mistral API with automatic fallback."""

import os

import httpx
from langchain_mistralai import ChatMistralAI
from langchain_ollama import ChatOllama

OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "mistral")
MISTRAL_MODEL = os.getenv("MISTRAL_MODEL", "mistral-small-latest")
# ollama | mistral | auto (Mistral when key set, Ollama fallback on rate limits)
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "auto").strip().lower()


def _ollama_llm(*, temperature: float, max_tokens: int | None):
    kwargs = {"model": OLLAMA_MODEL, "temperature": temperature}
    if max_tokens is not None:
        kwargs["num_predict"] = max_tokens
    return ChatOllama(**kwargs)


def _mistral_llm(*, temperature: float, max_tokens: int | None):
    kwargs = {
        "model": MISTRAL_MODEL,
        "mistral_api_key": os.environ["MISTRAL_API_KEY"],
        "temperature": temperature,
    }
    if max_tokens is not None:
        kwargs["max_tokens"] = max_tokens
    return ChatMistralAI(**kwargs)


def get_llm(*, temperature: float = 0.3, max_tokens: int | None = None):
    """
    Return the configured chat model.

    LLM_PROVIDER:
      - ollama  — always local Ollama (free, no rate limits)
      - mistral — always Mistral API (requires MISTRAL_API_KEY)
      - auto    — Mistral when key is set, with Ollama fallback on API errors
    """
    ollama = _ollama_llm(temperature=temperature, max_tokens=max_tokens)
    mistral_key = os.getenv("MISTRAL_API_KEY", "").strip()

    if LLM_PROVIDER == "ollama" or not mistral_key:
        if LLM_PROVIDER == "mistral" and not mistral_key:
            print("Warning: LLM_PROVIDER=mistral but MISTRAL_API_KEY is unset; using Ollama.")
        return ollama

    mistral = _mistral_llm(temperature=temperature, max_tokens=max_tokens)
    if LLM_PROVIDER == "mistral":
        return mistral

    return mistral.with_fallbacks(
        [ollama],
        exceptions_to_handle=(httpx.HTTPStatusError, httpx.ConnectError, httpx.TimeoutException),
    )
