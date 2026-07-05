from ..base import BaseAIProvider
from typing import Dict, Any, List


class GroqProvider(BaseAIProvider):
    """Groq cloud AI provider — ultra-fast inference (llama3-70b, mixtral, gemma2, etc.)

    Groq offers a generous free tier and is significantly faster than OpenAI
    for real-time use cases. Uses an OpenAI-compatible API.

    Requires the `groq` package: `pip install groq`
    Only dataset metadata (stats, null counts) is sent — never raw data rows.

    Usage:
        dp.analyze(df, use_ai=True, ai_provider="groq",
                   ai_model="llama3-70b-8192", api_key="gsk_...")

    Popular free models:
        - llama3-70b-8192    (recommended — best quality)
        - llama3-8b-8192     (fastest)
        - mixtral-8x7b-32768 (long context)
        - gemma2-9b-it       (Google's Gemma via Groq)
    """

    def generate(
        self,
        meta: Dict[str, Any],
        missing_list: List[Dict[str, Any]],
        strong_relations: List[str],
    ) -> str:
        try:
            from groq import Groq
        except ImportError:
            return (
                "⚠️  Groq package not installed. "
                "Run: pip install groq"
            )

        try:
            client = Groq(api_key=self.api_key)
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
            return f"⚠️  Groq API error: {e}"
