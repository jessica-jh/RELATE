"""Locally-hosted target models served via a vLLM OpenAI-compatible endpoint
(e.g. on the school GPU server).

Config: pass base_url in the target entry's extra kwargs, e.g.
  {provider: "local", model: "meta-llama/Llama-3.1-8B-Instruct", base_url: "http://127.0.0.1:8001/v1"}
"""
import os

from tenacity import retry, stop_after_attempt, wait_exponential

from .base import Provider

_DEFAULT_BASE_URL = "http://127.0.0.1:8000/v1"


class LocalProvider(Provider):
    def __init__(self, model, **kwargs):
        super().__init__(model, **kwargs)
        from openai import OpenAI
        base_url = kwargs.get("base_url") or os.environ.get("LOCAL_LLM_BASE_URL", _DEFAULT_BASE_URL)
        # vLLM's OpenAI server doesn't validate the key by default; any string works.
        self.client = OpenAI(api_key="local-no-auth", base_url=base_url)

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
