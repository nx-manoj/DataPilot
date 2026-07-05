from .factory import get_provider
from typing import Dict, Any, List, Optional


def generate_insights(
    meta: Dict[str, Any],
    missing_list: List[Dict[str, Any]],
    strong_relations: List[str],
    model_name: str = "llama3",
    ai_provider: str = "ollama",
    api_key: Optional[str] = None,
) -> str:
    """Sends lightweight dataset metadata to the configured AI provider and
    returns conversational, context-aware engineering recommendations.

    DataPilot uses a Metadata-Only AI Pattern — raw data rows are NEVER
    transmitted to any provider. Only concise statistical summaries are sent.

    Args:
        meta:             Structural stats dict from dp.summary().
        missing_list:     List of column null-stat dicts from dp.missing().
        strong_relations: List of correlation pair strings from dp.correlation().
        model_name:       Model to use within the chosen provider.
                          Defaults vary per provider if not specified.
        ai_provider:      AI backend to use. Options:
                            'ollama'  — local, free, private (default)
                            'openai'  — GPT-4o, GPT-4, GPT-3.5
                            'gemini'  — Google Gemini 1.5 Pro/Flash
                            'claude'  — Anthropic Claude 3.5 Sonnet/Haiku
                            'groq'    — Ultra-fast free-tier (Llama3, Mixtral)
        api_key:          API key for cloud providers. Not needed for Ollama.

    Returns:
        Plain-text string with 3 actionable data science recommendations.
    """
    try:
        provider = get_provider(
            ai_provider=ai_provider,
            ai_model=model_name,
            api_key=api_key,
        )
        return provider.generate(meta, missing_list, strong_relations)
    except ValueError as e:
        return f"⚠️  Configuration error: {e}"
    except Exception as e:
        return f"⚠️  Could not generate AI Insights. Error: {e}"
