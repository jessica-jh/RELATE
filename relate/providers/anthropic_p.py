"""Anthropic (Claude Haiku) validation judge.

Requires:  pip install anthropic
Env:       ANTHROPIC_API_KEY
"""
import os

from tenacity import retry, stop_after_attempt, wait_exponential

from .base import Provider


class AnthropicProvider(Provider):
    def __init__(self, model, **kwargs):
        super().__init__(model, **kwargs)
        import anthropic
        key = os.environ.get("ANTHROPIC_API_KEY")
        if not key:
            raise RuntimeError("ANTHROPIC_API_KEY not set")
        self.client = anthropic.Anthropic(api_key=key)

    @retry(stop=stop_after_attempt(4),
           wait=wait_exponential(multiplier=1, min=2, max=30))
    def chat(self, messages, *, system=None, temperature=0.0, max_tokens=1024, seed=None):
        # Anthropic's API has no seed parameter; accepted for interface parity only.
        resp = self.client.messages.create(
            model=self.model,
            system=system or "",
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return "".join(b.text for b in resp.content if b.type == "text").strip()
