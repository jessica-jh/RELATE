"""Gemini Flash primary judge.

Requires:  pip install google-genai
Env:       GEMINI_API_KEY  (or GOOGLE_API_KEY)
"""
import os

from tenacity import retry, stop_after_attempt, wait_exponential

from .base import Provider


class GeminiProvider(Provider):
    def __init__(self, model, **kwargs):
        super().__init__(model, **kwargs)
        from google import genai
        key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        if not key:
            raise RuntimeError("GEMINI_API_KEY / GOOGLE_API_KEY not set")
        self.genai = genai
        self.client = genai.Client(api_key=key)

    @retry(stop=stop_after_attempt(4),
           wait=wait_exponential(multiplier=1, min=2, max=30))
    def chat(self, messages, *, system=None, temperature=0.0, max_tokens=1024, seed=None):
        # flatten chat history; system goes in system_instruction
        parts = []
        for m in messages:
            role = m["role"].upper()
            parts.append(f"{role}: {m['content']}")
        prompt = "\n\n".join(parts)
        cfg = self.genai.types.GenerateContentConfig(
            temperature=temperature,
            max_output_tokens=max_tokens,
            system_instruction=system,
            seed=seed,
            # disable thinking so hidden tokens don't eat max_output_tokens
            thinking_config=self.genai.types.ThinkingConfig(thinking_budget=0),
        )
        resp = self.client.models.generate_content(
            model=self.model, contents=prompt, config=cfg,
        )
        return (resp.text or "").strip()
