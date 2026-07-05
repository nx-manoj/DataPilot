from .base import BaseAIProvider
from typing import Optional


# Registry mapping provider name → class (lazy import avoids hard dependencies)
_PROVIDER_REGISTRY = {
    "ollama": "datapilot.ai.providers.ollama_provider.OllamaProvider",
    "openai": "datapilot.ai.providers.openai_provider.OpenAIProvider",
    "gemini": "datapilot.ai.providers.gemini_provider.GeminiProvider",
    "claude": "datapilot.ai.providers.claude_provider.ClaudeProvider",
    "groq":   "datapilot.ai.providers.groq_provider.GroqProvider",
}

# Human-readable default models per provider
DEFAULT_MODELS = {
    "ollama": "llama3",
    "openai": "gpt-4o-mini",
    "gemini": "gemini-1.5-flash",
    "claude": "claude-3-haiku-20240307",
    "groq":   "llama3-70b-8192",
}


def get_provider(
    ai_provider: str,
    ai_model: Optional[str] = None,
    api_key: Optional[str] = None,
) -> BaseAIProvider:
    """Resolves and instantiates the correct AI provider from a string name.

    Args:
        ai_provider: Name of the AI backend. One of:
                     'ollama' (local, default), 'openai', 'gemini', 'claude', 'groq'.
        ai_model:    Model name within the provider. If None, uses the provider's
                     sensible default (e.g. 'gpt-4o-mini' for OpenAI).
        api_key:     API key required for cloud providers. Not needed for Ollama.

    Returns:
        An instantiated BaseAIProvider subclass ready to call .generate().

    Raises:
        ValueError: If the provider name is not recognised.
    """
    provider_name = ai_provider.lower().strip()

    if provider_name not in _PROVIDER_REGISTRY:
        supported = ", ".join(f"'{k}'" for k in _PROVIDER_REGISTRY)
        raise ValueError(
            f"Unknown AI provider: '{ai_provider}'. "
            f"Supported providers: {supported}."
        )

    # Validate API key requirement for cloud providers
    if provider_name != "ollama" and not api_key:
        raise ValueError(
            f"'{ai_provider}' is a cloud provider and requires an API key. "
            f"Pass it via: dp.analyze(df, use_ai=True, ai_provider='{ai_provider}', "
            f"api_key='your-key-here')"
        )

    # Resolve default model
    model = ai_model or DEFAULT_MODELS[provider_name]

    # Lazy import of the provider class
    module_path, class_name = _PROVIDER_REGISTRY[provider_name].rsplit(".", 1)
    import importlib
    module = importlib.import_module(module_path)
    provider_class = getattr(module, class_name)

    return provider_class(model=model, api_key=api_key)
