from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional


class BaseAIProvider(ABC):
    """Abstract base class for all DataPilot AI provider backends.

    Every provider must implement a single `generate` method that accepts
    the dataset metadata and returns a plain-text insight string.
    Raw data rows are NEVER passed to any provider — only lightweight
    statistical summaries are transmitted (metadata-only pattern).
    """

    def __init__(self, model: str, api_key: str = None, **kwargs):
        self.model = model
        self.api_key = api_key

    @abstractmethod
    def generate(
        self,
        meta: Dict[str, Any],
        missing_list: List[Dict[str, Any]],
        strong_relations: List[str],
    ) -> str:
        """Generate AI insights from dataset metadata.

        Args:
            meta: Structural stats dict (rows, columns, dtypes, null counts).
            missing_list: List of dicts with column null stats.
            strong_relations: List of strong correlation pair strings.

        Returns:
            A plain-text string with actionable recommendations.
        """

    def _build_system_prompt(self) -> str:
        return (
            "You are DataPilot, an elite data science assistant. "
            "Analyse the structural statistics provided and write exactly 3 brief, "
            "impactful bullet points giving expert data preprocessing, cleaning, or "
            "modelling advice. Be direct and conversational. "
            "Do not include introductory text or markdown headers."
        )

    def _build_user_prompt(
        self,
        meta: Dict[str, Any],
        missing_list: List[Dict[str, Any]],
        strong_relations: List[str],
    ) -> str:
        return (
            f"Dataset Profile:\n"
            f"- Dimensions: {meta['rows']} rows × {meta['columns']} columns\n"
            f"- Duplicate Rows: {meta.get('duplicates_count', 'N/A')}\n"
            f"- Missing Data: {missing_list}\n"
            f"- Strong Correlations: {strong_relations}\n"
        )

    def _call_with_raw_prompts(
        self,
        system_prompt: str,
        user_prompt: str,
    ) -> str:
        """Send arbitrary system + user prompts to the provider.

        Used by dp.ask_ai() and dp.visualize_ai() for free-form queries.
        Subclasses should override this for direct access to the chat API;
        the default falls back to the structured generate() interface.

        Args:
            system_prompt: The system/instruction message.
            user_prompt:   The user message (typically includes dataset context).

        Returns:
            Plain-text AI response string.
        """
        # Default fallback: pack prompts into the generate() interface
        meta = {"rows": 0, "columns": 0, "duplicates_count": "N/A"}
        return self.generate(meta, [], [user_prompt])
