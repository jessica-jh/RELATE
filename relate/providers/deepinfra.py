"""DeepInfra target models via the OpenAI-compatible endpoint.

Requires:  pip install openai
Env:       DEEPINFRA_API_KEY
"""
import os

from tenacity import retry, stop_after_attempt, wait_exponential

from .base import Provider

_BASE_URL = "https://api.deepinfra.com/v1/openai"


class DeepInfraProvider(Provider):
    def __init__(self, model, **kwargs):
        super().__init__(model, **kwargs)
        from openai import OpenAI
        key = os.environ.get("DEEPINFRA_API_KEY")
        if not key:
            raise RuntimeError("DEEPINFRA_API_KEY not set")
        self.client = OpenAI(api_key=key, base_url=_BASE_URL)

    @retry(stop=stop_after_attempt(4),
           wait=wait_exponential(multiplier=1, min=2, max=30))
    def chat(self, messages, *, system=None, temperature=0.0, max_tokens=1024, seed=None):
        msgs = ([{"role": "system", "content": system}] if system else []) + messages
        extra_body = self.kwargs.get("extra_body")
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=msgs,
            temperature=temperature,
            max_tokens=max_tokens,
            **({"seed": seed} if seed is not None else {}),
            **({"extra_body": extra_body} if extra_body else {}),
        )
        return (resp.choices[0].message.content or "").strip()
