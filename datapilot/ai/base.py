from abc import ABC, abstractmethod
from typing import Dict, Any, List


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
