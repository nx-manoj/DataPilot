from ..base import BaseAIProvider
from typing import Dict, Any, List


class GeminiProvider(BaseAIProvider):
    """Google Gemini cloud AI provider (gemini-1.5-pro, gemini-1.5-flash, etc.)

    Requires the `google-generativeai` package: `pip install google-generativeai`
    Only dataset metadata (stats, null counts) is sent — never raw data rows.

    Usage:
        dp.analyze(df, use_ai=True, ai_provider="gemini",
                   ai_model="gemini-1.5-flash", api_key="AIza...")
    """

    def generate(
        self,
        meta: Dict[str, Any],
        missing_list: List[Dict[str, Any]],
        strong_relations: List[str],
    ) -> str:
        try:
            import google.generativeai as genai
        except ImportError:
            return (
                "⚠️  Google Generative AI package not installed. "
                "Run: pip install google-generativeai"
            )

        try:
            genai.configure(api_key=self.api_key)
            model = genai.GenerativeModel(
                model_name=self.model,
                system_instruction=self._build_system_prompt(),
            )
            user_prompt = self._build_user_prompt(meta, missing_list, strong_relations)
            response = model.generate_content(
                user_prompt,
                generation_config=genai.GenerationConfig(
                    max_output_tokens=300,
                    temperature=0.4,
                ),
            )
            return response.text.strip()
        except Exception as e:
            return f"⚠️  Google Gemini API error: {e}"

    def _call_with_raw_prompts(self, system_prompt: str, user_prompt: str) -> str:
        try:
            import google.generativeai as genai
        except ImportError:
            return "⚠️  Google Generative AI package not installed. Run: pip install google-generativeai"
        try:
            genai.configure(api_key=self.api_key)
            model = genai.GenerativeModel(
                model_name=self.model,
                system_instruction=system_prompt,
            )
            response = model.generate_content(
                user_prompt,
                generation_config=genai.GenerationConfig(
                    max_output_tokens=500,
                    temperature=0.4,
                ),
            )
            return response.text.strip()
        except Exception as e:
            return f"⚠️  Google Gemini API error: {e}"
