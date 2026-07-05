from ..base import BaseAIProvider
from typing import Dict, Any, List


class ClaudeProvider(BaseAIProvider):
    """Anthropic Claude cloud AI provider (claude-3-5-sonnet, claude-3-haiku, etc.)

    Requires the `anthropic` package: `pip install anthropic`
    Only dataset metadata (stats, null counts) is sent — never raw data rows.

    Usage:
        dp.analyze(df, use_ai=True, ai_provider="claude",
                   ai_model="claude-3-5-sonnet-20241022", api_key="sk-ant-...")
    """

    def generate(
        self,
        meta: Dict[str, Any],
        missing_list: List[Dict[str, Any]],
        strong_relations: List[str],
    ) -> str:
        try:
            import anthropic
        except ImportError:
            return (
                "⚠️  Anthropic package not installed. "
                "Run: pip install anthropic"
            )

        try:
            client = anthropic.Anthropic(api_key=self.api_key)
            response = client.messages.create(
                model=self.model,
                max_tokens=300,
                system=self._build_system_prompt(),
                messages=[
                    {
                        "role": "user",
                        "content": self._build_user_prompt(meta, missing_list, strong_relations),
                    }
                ],
            )
            return response.content[0].text.strip()
        except Exception as e:
            return f"⚠️  Anthropic Claude API error: {e}"
