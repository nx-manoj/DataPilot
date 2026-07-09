from ..base import BaseAIProvider
from typing import Dict, Any, List


class OpenAIProvider(BaseAIProvider):
    """OpenAI cloud AI provider (GPT-4o, GPT-4, GPT-3.5-turbo, etc.)

    Requires the `openai` package: `pip install openai`
    Only dataset metadata (stats, null counts) is sent — never raw data rows.

    Usage:
        dp.analyze(df, use_ai=True, ai_provider="openai",
                   ai_model="gpt-4o", api_key="sk-...")
    """

    def generate(
        self,
        meta: Dict[str, Any],
        missing_list: List[Dict[str, Any]],
        strong_relations: List[str],
    ) -> str:
        try:
            from openai import OpenAI
        except ImportError:
            return (
                "⚠️  OpenAI package not installed. "
                "Run: pip install openai"
            )

        try:
            client = OpenAI(api_key=self.api_key)
            response = client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self._build_system_prompt()},
                    {"role": "user",   "content": self._build_user_prompt(meta, missing_list, strong_relations)},
                ],
                max_tokens=300,
                temperature=0.4,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            return f"⚠️  OpenAI API error: {e}"

    def _call_with_raw_prompts(self, system_prompt: str, user_prompt: str) -> str:
        try:
            from openai import OpenAI
        except ImportError:
            return "⚠️  OpenAI package not installed. Run: pip install openai"
        try:
            client = OpenAI(api_key=self.api_key)
            response = client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user",   "content": user_prompt},
                ],
                max_tokens=500,
                temperature=0.4,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            return f"⚠️  OpenAI API error: {e}"
