"""Provider abstraction: chat(messages) -> reply string.

Only this layer talks to specific SDKs.
"""
from abc import ABC, abstractmethod
from typing import Any


class Provider(ABC):
    def __init__(self, model: str, **kwargs: Any) -> None:
        self.model = model
        self.kwargs = kwargs

    @abstractmethod
    def chat(
        self,
        messages: list[dict[str, str]],
        *,
        system: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 1024,
        seed: int | None = None,
    ) -> str:
        """Return the assistant's reply text for the given messages.

        `seed` is best-effort: providers that support deterministic sampling
        should use it for reproducibility; providers that don't may ignore it.
        """
        raise NotImplementedError


def get_provider(provider: str, model: str, **kwargs: Any) -> Provider:
    """Lazy factory so missing SDKs/keys don't break mock runs."""
    provider = provider.lower()
    if provider == "mock":
        from .mock import MockProvider
        return MockProvider(model, **kwargs)
    if provider == "deepinfra":
        from .deepinfra import DeepInfraProvider
        return DeepInfraProvider(model, **kwargs)
    if provider == "gemini":
        from .gemini import GeminiProvider
        return GeminiProvider(model, **kwargs)
    if provider == "anthropic":
        from .anthropic_p import AnthropicProvider
        return AnthropicProvider(model, **kwargs)
    if provider == "openai":
        from .openai_p import OpenAIProvider
        return OpenAIProvider(model, **kwargs)
    if provider == "local":
        from .local import LocalProvider
        return LocalProvider(model, **kwargs)
    raise ValueError(f"Unknown provider: {provider!r}")
