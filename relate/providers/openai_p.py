"""OpenAI models (e.g. gpt-4o-mini) for the user-agent / stimulus writer.

Requires:  pip install openai
Env:       OPENAI_API_KEY
"""
import os

from tenacity import retry, stop_after_attempt, wait_exponential

from .base import Provider


class OpenAIProvider(Provider):
    def __init__(self, model, **kwargs):
        super().__init__(model, **kwargs)
        from openai import OpenAI
        key = os.environ.get("OPENAI_API_KEY")
        if not key:
            raise RuntimeError("OPENAI_API_KEY not set")
        self.client = OpenAI(api_key=key)

    @retry(stop=stop_after_attempt(4),
           wait=wait_exponential(multiplier=1, min=2, max=30))
    def chat(self, messages, *, system=None, temperature=0.0, max_tokens=1024, seed=None):
        msgs = ([{"role": "system", "content": system}] if system else []) + messages
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=msgs,
            temperature=temperature,
            max_tokens=max_tokens,
            **({"seed": seed} if seed is not None else {}),
        )
        return (resp.choices[0].message.content or "").strip()
