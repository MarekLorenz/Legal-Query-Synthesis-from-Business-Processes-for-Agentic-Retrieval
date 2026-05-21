from __future__ import annotations

import os
from abc import ABC, abstractmethod

import requests


class LLMProvider(ABC):
    """Abstract base class for LLM providers."""

    @abstractmethod
    def generate(self, prompt: str, system_instruction: str | None = None) -> str:
        """Generate text from a prompt (optional system instruction for Gemini-style APIs)."""
        pass


class OllamaProvider(LLMProvider):
    """Local Ollama LLM provider."""

    def __init__(self, model: str = "deepseek-r1:latest", base_url: str = "http://localhost:11434"):
        self.model = model
        self.base_url = base_url
        self.url = f"{self.base_url}/api/generate"

    def generate(self, prompt: str, system_instruction: str | None = None) -> str:
        """Generate text using Ollama local model."""
        full_prompt = prompt
        if system_instruction:
            full_prompt = f"{system_instruction.strip()}\n\n{prompt}"
        try:
            payload = {"model": self.model, "prompt": full_prompt, "stream": False}
            response = requests.post(self.url, json=payload, timeout=120)
            response.raise_for_status()
            return response.json().get("response", "").strip()
        except requests.exceptions.ConnectionError:
            print(f"[ERROR] Ollama connection failed at {self.url}")
            return ""
        except Exception as e:
            print(f"[ERROR] Ollama generation failed: {e}")
            return ""


DEFAULT_GEMINI_SYSTEM = (
    "You are a precise assistant. Follow the user instructions exactly and return only what is asked."
)


class GeminiProvider(LLMProvider):
    """Google Gemini API via google-genai (Client + GenerateContentConfig + system_instruction)."""

    def __init__(
        self,
        model: str = "gemini-3.1-pro-preview",
        thinking_budget: int = 2048,
        tokens_log_path: str | None = "tokens",
    ):
        try:
            from dotenv import load_dotenv
            from google import genai

            load_dotenv()
            api_key = os.environ.get("GOOGLE_API_KEY")
            if not api_key:
                raise ValueError("GOOGLE_API_KEY is not set in the environment")
            self.client = genai.Client(api_key=api_key)
            self.model = model
            self.thinking_budget = thinking_budget
            self.tokens_log_path = tokens_log_path
        except ImportError as e:
            raise ImportError("Gemini provider requires google-genai and python-dotenv packages") from e

    def _append_token_count(self, response) -> None:
        if not self.tokens_log_path:
            return
        um = getattr(response, "usage_metadata", None)
        if um is None:
            return
        candidates = getattr(um, "candidates_token_count", None) or 0
        thoughts = getattr(um, "thoughts_token_count", None) or 0
        output_tokens = int(candidates) + int(thoughts)
        try:
            with open(self.tokens_log_path, "a", encoding="utf-8") as f:
                f.write(f"{output_tokens}\n")
        except OSError:
            pass

    def generate(self, prompt: str, system_instruction: str | None = None) -> str:
        try:
            from google.genai import types

            system = (system_instruction or DEFAULT_GEMINI_SYSTEM).strip()
            response = self.client.models.generate_content(
                model=self.model,
                contents=prompt,
                config=types.GenerateContentConfig(
                    thinking_config=types.ThinkingConfig(thinking_budget=self.thinking_budget),
                    system_instruction=system,
                ),
            )
            self._append_token_count(response)
            text = getattr(response, "text", None) or ""
            return text.strip()
        except Exception as e:
            print(f"[ERROR] Gemini generation failed: {e}")
            return ""


def get_llm_provider(provider: str = "ollama", **kwargs) -> LLMProvider:
    """
    Factory function to get LLM provider.

    Args:
        provider: "ollama" (default) or "gemini"
        **kwargs: Additional arguments passed to provider

    Returns:
        LLMProvider instance
    """
    if provider.lower() == "gemini":
        return GeminiProvider(**kwargs)
    if provider.lower() == "ollama":
        return OllamaProvider(**kwargs)
    raise ValueError(f"Unknown provider: {provider}. Use 'ollama' or 'gemini'")
