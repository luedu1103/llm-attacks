"""Shared Ollama client using the OpenAI-compatible endpoint."""

from __future__ import annotations

import logging
import subprocess
import time
from functools import wraps
from typing import Callable, TypeVar

import httpx
from openai import OpenAI

import config

logger = logging.getLogger(__name__)

_client: OpenAI | None = None

F = TypeVar("F", bound=Callable)

MAX_RETRIES = 3
RETRY_BASE_WAIT = 2.0


def _with_retry(fn: F) -> F:
    """Decorator: retry with exponential backoff on any exception."""

    @wraps(fn)
    def wrapper(*args, **kwargs):
        last_exc: Exception | None = None
        for attempt in range(MAX_RETRIES):
            try:
                return fn(*args, **kwargs)
            except Exception as e:
                last_exc = e
                wait = RETRY_BASE_WAIT * (2**attempt)  # 2s, 4s, 8s
                logger.warning(
                    "chat() failed (attempt %d/%d): %s — retrying in %.0fs",
                    attempt + 1,
                    MAX_RETRIES,
                    e,
                    wait,
                )
                time.sleep(wait)
        raise RuntimeError(f"chat() failed after {MAX_RETRIES} attempts") from last_exc

    return wrapper


def _ollama_is_running() -> bool:
    try:
        httpx.get("http://localhost:11434", timeout=2)
        return True
    except Exception:
        return False


def ensure_ollama() -> None:
    """Start Ollama in the background if it is not already running."""
    if _ollama_is_running():
        return
    logger.info("Starting Ollama...")
    subprocess.Popen(
        ["ollama", "serve"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    for _ in range(10):
        time.sleep(1)
        if _ollama_is_running():
            logger.info("Ollama ready.")
            return
    raise RuntimeError("Ollama did not start in time.")


def get_client() -> OpenAI:
    global _client
    if _client is None:
        ensure_ollama()
        _client = OpenAI(base_url=config.OLLAMA_BASE_URL, api_key="ollama")
    return _client


@_with_retry
def chat(prompt: str, system: str = "") -> str:
    """Send a prompt to Ollama and return the response text.

    Automatically retries up to MAX_RETRIES times with exponential backoff
    if Ollama is temporarily unavailable or returns an error.
    """
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    response = get_client().chat.completions.create(
        model=config.OLLAMA_MODEL,
        messages=messages,
        temperature=0.7,
    )
    return response.choices[0].message.content.strip()
