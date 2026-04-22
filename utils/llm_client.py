"""Shared Ollama client using the OpenAI-compatible endpoint."""

from __future__ import annotations

import subprocess
import time

import httpx
from openai import OpenAI
import config

_client: OpenAI | None = None


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
    print("Starting Ollama...")
    subprocess.Popen(
        ["ollama", "serve"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    for _ in range(10):
        time.sleep(1)
        if _ollama_is_running():
            print("Ollama ready.")
            return
    raise RuntimeError("Ollama did not start in time.")


def get_client() -> OpenAI:
    global _client
    if _client is None:
        ensure_ollama()
        _client = OpenAI(base_url=config.OLLAMA_BASE_URL, api_key="ollama")
    return _client


def chat(prompt: str, system: str = "") -> str:
    """Send a prompt to Ollama and return the response text."""
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
