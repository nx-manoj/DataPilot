"""
datapilot.config — Global session configuration singleton.

Call dp.configure() once at the start of your notebook/script to set
the AI provider, model, and API key for the entire session.  Every
function that accepts use_ai=True will automatically pick up these
settings — no need to pass the key on every call.

Example
-------
    import datapilot as dp

    dp.configure(ai_provider="groq", api_key="gsk_...", ai_model="llama3-70b-8192")

    dp.analyze(df, use_ai=True)       # uses groq + stored key
    dp.suggest(df, use_ai=True)       # same
    dp.compare(train, test, use_ai=True)
"""

from typing import Optional

# ── Internal singleton dict ───────────────────────────────────────────────────
_cfg: dict = {
    "ai_provider": "ollama",
    "ai_model":    None,
    "api_key":     None,
}


def configure(
    ai_provider: str = "ollama",
    ai_model: Optional[str] = None,
    api_key: Optional[str] = None,
) -> None:
    """Set the global AI provider, model, and API key for the session.

    Args:
        ai_provider: One of 'ollama' (local/default), 'openai', 'gemini',
                     'claude', 'groq'.
        ai_model:    Model name to use within the provider.  Falls back to the
                     provider's sensible default when None.
        api_key:     API key for cloud providers. Not needed for Ollama.
    """
    _cfg["ai_provider"] = ai_provider
    _cfg["ai_model"]    = ai_model
    _cfg["api_key"]     = api_key
    print(
        f"✅ DataPilot configured — provider: '{ai_provider}'"
        + (f" | model: '{ai_model}'" if ai_model else "")
        + (" | api_key: [set]" if api_key else " | api_key: [not set]")
    )


def get_config() -> dict:
    """Return a shallow copy of the current global configuration."""
    return _cfg.copy()
