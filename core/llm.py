"""Shared LLM factory: Mistral API when key is set, Ollama locally otherwise."""

import os

from langchain_mistralai import ChatMistralAI
from langchain_ollama import ChatOllama


OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "mistral")
MISTRAL_MODEL = os.getenv("MISTRAL_MODEL", "mistral-small-latest")


def get_llm(*, temperature: float = 0.3, max_tokens: int | None = None):
    """
    Return ChatMistralAI if MISTRAL_API_KEY is set, else ChatOllama (free, local M4).
    """
    mistral_key = os.getenv("MISTRAL_API_KEY")
    if mistral_key:
        kwargs = {
            "model": MISTRAL_MODEL,
            "mistral_api_key": mistral_key,
            "temperature": temperature,
        }
        if max_tokens is not None:
            kwargs["max_tokens"] = max_tokens
        return ChatMistralAI(**kwargs)

    kwargs = {
        "model": OLLAMA_MODEL,
        "temperature": temperature,
    }
    if max_tokens is not None:
        kwargs["num_predict"] = max_tokens
    return ChatOllama(**kwargs)
