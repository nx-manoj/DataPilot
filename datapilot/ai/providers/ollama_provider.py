from ..base import BaseAIProvider
from typing import Dict, Any, List


class OllamaProvider(BaseAIProvider):
    """Local Ollama AI provider — fully private, no internet required.

    Communicates with a locally running Ollama daemon. Raw data rows
    are never transmitted; only metadata summaries are sent.

    Usage:
        dp.analyze(df, use_ai=True, ai_provider="ollama", ai_model="llama3")
    """

    def generate(
        self,
        meta: Dict[str, Any],
        missing_list: List[Dict[str, Any]],
        strong_relations: List[str],
    ) -> str:
        try:
            import ollama
            response = ollama.chat(
                model=self.model,
                messages=[
                    {"role": "system", "content": self._build_system_prompt()},
                    {"role": "user",   "content": self._build_user_prompt(meta, missing_list, strong_relations)},
                ],
            )
            return response["message"]["content"].strip()
        except Exception as e:
            return (
                f"⚠️  Could not reach Ollama. Ensure it is running (`ollama serve`). "
                f"Error: {e}"
            )
