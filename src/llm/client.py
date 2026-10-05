"""
LLM Provider Abstraction Layer, Rate Limiter, and Caching Layer.

Provides:
- BaseLLMProvider interface
- OpenAIProvider (GPT-4o, GPT-4o-mini, etc.)
- GeminiProvider (Google Gemini API via REST/google-genai)
- AnthropicProvider (Claude 3.5 Sonnet, etc.)
- LocalProvider (Ollama)
- MockProvider (deterministic, zero-cost mock for tests and offline development)
- Factory get_llm_provider()
"""

import os
import sys
import json
import time
import hashlib
import re
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

from src.utils.common import DATA_CACHE
from src.utils.logging import get_logger
from src.utils.config import load_config

logger = get_logger(__name__)


def load_dotenv_if_available() -> None:
    """Load local .env values without requiring python-dotenv at import time."""
    try:
        from dotenv import load_dotenv

        load_dotenv()
        return
    except ImportError:
        pass

    env_path = Path.cwd() / ".env"
    if not env_path.exists():
        env_path = Path(__file__).resolve().parents[2] / ".env"
    if not env_path.exists():
        return

    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        os.environ.setdefault(key, value)


class BaseLLMProvider(ABC):
    """Abstract base class for all LLM providers."""

    def __init__(
        self,
        model_name: str,
        temperature: float = 0.5,
        max_tokens: int = 512,
        cache_enabled: bool = True,
        cache_dir: Optional[Path] = None,
        max_retries: int = 3,
        retry_delay: float = 2.0,
    ):
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.cache_enabled = cache_enabled
        self.cache_dir = cache_dir or (DATA_CACHE / "llm")
        self.max_retries = max_retries
        self.retry_delay = retry_delay

        if self.cache_enabled:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self.total_cost_usd = 0.0

    def _get_cache_key(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """Deterministic cache key incorporating model name, temperature, and prompt content."""
        content = f"{self.model_name}|{self.temperature}|{system_prompt or ''}|{prompt}"
        return hashlib.sha256(content.encode("utf-8")).hexdigest()

    def _check_cache(self, cache_key: str) -> Optional[str]:
        """Check disk cache for previously generated output."""
        if not self.cache_enabled:
            return None
        cache_file = self.cache_dir / f"{cache_key}.json"
        if cache_file.exists():
            try:
                with open(cache_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    return data.get("response")
            except Exception:
                return None
        return None

    def _save_cache(
        self,
        cache_key: str,
        prompt: str,
        response: str,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """Persist output to disk cache."""
        if not self.cache_enabled:
            return
        cache_file = self.cache_dir / f"{cache_key}.json"
        try:
            with open(cache_file, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "model": self.model_name,
                        "temperature": self.temperature,
                        "prompt": prompt,
                        "response": response,
                        "metadata": metadata or {},
                        "timestamp": time.time(),
                    },
                    f,
                    indent=2,
                    ensure_ascii=False,
                )
        except Exception as e:
            logger.warning(f"Failed to write cache {cache_key}: {e}")

    @abstractmethod
    def _call_api(
        self, prompt: str, system_prompt: Optional[str] = None
    ) -> Tuple[str, int, int]:
        """
        Provider-specific call.
        Returns: (response_text, prompt_tokens, completion_tokens)
        """
        pass

    def generate(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """Generate response with disk caching and exponential backoff retries."""
        key = self._get_cache_key(prompt, system_prompt)
        cached = self._check_cache(key)
        if cached is not None:
            return cached

        delay = self.retry_delay
        last_exception = None

        for attempt in range(1, self.max_retries + 1):
            try:
                response_text, p_toks, c_toks = self._call_api(prompt, system_prompt)
                self.total_prompt_tokens += p_toks
                self.total_completion_tokens += c_toks
                self.total_cost_usd += self.estimate_cost(p_toks, c_toks)

                self._save_cache(
                    key,
                    prompt,
                    response_text,
                    {"prompt_tokens": p_toks, "completion_tokens": c_toks},
                )
                return response_text
            except Exception as e:
                last_exception = e
                logger.warning(
                    f"LLM API call failed (attempt {attempt}/{self.max_retries}): {e}. Retrying in {delay:.1f}s..."
                )
                time.sleep(delay)
                delay *= 2

        raise RuntimeError(
            f"LLM generation failed after {self.max_retries} attempts: {last_exception}"
        )

    def batch_generate(
        self, prompts: List[str], system_prompt: Optional[str] = None
    ) -> List[str]:
        """Generate responses sequentially with caching and progress logging."""
        responses = []
        for i, p in enumerate(prompts):
            resp = self.generate(p, system_prompt)
            responses.append(resp)
            if (i + 1) % 10 == 0 or i == len(prompts) - 1:
                logger.info(f"Generated {i + 1}/{len(prompts)} responses...")
        return responses

    @abstractmethod
    def estimate_cost(self, prompt_tokens: int, completion_tokens: int) -> float:
        """Estimate USD cost based on token counts."""
        pass


class OpenAIProvider(BaseLLMProvider):
    """OpenAI API provider implementation."""

    def __init__(
        self, model_name: str = "gpt-4o-mini", api_key: Optional[str] = None, **kwargs
    ):
        super().__init__(model_name=model_name, **kwargs)
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY is not set.")
        from openai import OpenAI
        self.client = OpenAI(api_key=self.api_key)

    def _call_api(
        self, prompt: str, system_prompt: Optional[str] = None
    ) -> Tuple[str, int, int]:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        choice = response.choices[0].message.content or ""
        p_toks = response.usage.prompt_tokens if response.usage else len(prompt.split())
        c_toks = (
            response.usage.completion_tokens if response.usage else len(choice.split())
        )
        return choice, p_toks, c_toks

    def estimate_cost(self, prompt_tokens: int, completion_tokens: int) -> float:
        # GPT-4o-mini pricing approx: $0.15/1M input, $0.60/1M output
        return (prompt_tokens * 0.00000015) + (completion_tokens * 0.00000060)


class GeminiProvider(BaseLLMProvider):
    """Google Gemini API provider implementation."""

    def __init__(
        self, model_name: str = "gemini-1.5-flash", api_key: Optional[str] = None, **kwargs
    ):
        super().__init__(model_name=model_name, **kwargs)
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY is not set.")

    def _call_api(
        self, prompt: str, system_prompt: Optional[str] = None
    ) -> Tuple[str, int, int]:
        import urllib.request
        full_prompt = f"{system_prompt}\n\n{prompt}" if system_prompt else prompt
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model_name}:generateContent?key={self.api_key}"

        payload = {
            "contents": [{"parts": [{"text": full_prompt}]}],
            "generationConfig": {
                "temperature": self.temperature,
                "maxOutputTokens": self.max_tokens,
            },
        }

        req = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        with urllib.request.urlopen(req, timeout=60) as resp:
            data = json.loads(resp.read().decode("utf-8"))

        cand = data.get("candidates", [{}])[0]
        parts = cand.get("content", {}).get("parts", [{}])
        text = parts[0].get("text", "")
        usage = data.get("usageMetadata", {})
        p_toks = usage.get("promptTokenCount", len(full_prompt.split()))
        c_toks = usage.get("candidatesTokenCount", len(text.split()))
        return text, p_toks, c_toks

    def estimate_cost(self, prompt_tokens: int, completion_tokens: int) -> float:
        return (prompt_tokens * 0.000000075) + (completion_tokens * 0.00000030)


class AnthropicProvider(BaseLLMProvider):
    """Anthropic Claude API provider implementation."""

    def __init__(
        self,
        model_name: str = "claude-3-5-sonnet-20241022",
        api_key: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(model_name=model_name, **kwargs)
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise ValueError("ANTHROPIC_API_KEY is not set.")
        from anthropic import Anthropic
        self.client = Anthropic(api_key=self.api_key)

    def _call_api(
        self, prompt: str, system_prompt: Optional[str] = None
    ) -> Tuple[str, int, int]:
        kwargs: Dict[str, Any] = {
            "model": self.model_name,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "messages": [{"role": "user", "content": prompt}],
        }
        if system_prompt:
            kwargs["system"] = system_prompt

        response = self.client.messages.create(**kwargs)
        text = response.content[0].text if response.content else ""
        p_toks = response.usage.input_tokens
        c_toks = response.usage.output_tokens
        return text, p_toks, c_toks

    def estimate_cost(self, prompt_tokens: int, completion_tokens: int) -> float:
        return (prompt_tokens * 0.000003) + (completion_tokens * 0.000015)


class LocalProvider(BaseLLMProvider):
    """Local Ollama provider implementation."""

    def __init__(
        self,
        model_name: str = "llama3",
        base_url: str = "http://localhost:11434",
        **kwargs,
    ):
        super().__init__(model_name=model_name, **kwargs)
        self.base_url = base_url.rstrip("/")

    def _call_api(
        self, prompt: str, system_prompt: Optional[str] = None
    ) -> Tuple[str, int, int]:
        import urllib.request
        url = f"{self.base_url}/api/generate"
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "system": system_prompt or "",
            "stream": False,
            "options": {
                "temperature": self.temperature,
                "num_predict": self.max_tokens,
            },
        }
        req = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=90) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        text = data.get("response", "")
        p_toks = data.get("prompt_eval_count", len(prompt.split()))
        c_toks = data.get("eval_count", len(text.split()))
        return text, p_toks, c_toks

    def estimate_cost(self, prompt_tokens: int, completion_tokens: int) -> float:
        return 0.0


class MockProvider(BaseLLMProvider):
    """
    High-fidelity heuristic mock LLM provider for zero-cost offline testing,
    smoke tests, unit tests, and CI/CD validation.
    """

    def __init__(self, model_name: str = "mock-llm-v1", **kwargs):
        super().__init__(model_name=model_name, **kwargs)

    def estimate_cost(self, prompt_tokens: int, completion_tokens: int) -> float:
        return 0.0

    def _call_api(
        self, prompt: str, system_prompt: Optional[str] = None
    ) -> Tuple[str, int, int]:
        prompt_lower = prompt.lower()

        # Deterministic financial classification based on domain terms
        bullish_cues = [
            "calls", "moon", "breakout", "strong", "growth", "bullish", "beat",
            "undervalued", "rocket", "buy", "upgraded", "long", "ath",
        ]
        bearish_cues = [
            "puts", "cut", "guidance cut", "downgrade", "tank", "bearish", "loss",
            "drill", "miss", "dilution", "bankrupt", "short", "dump",
        ]

        # Path variation logic: simulate temperature sampling across paths.
        # The labeling module appends markers such as "[Sampling Path 7/8]".
        path_offset = 0
        path_match = re.search(r"\[(?:sampling\s+)?path\s+(\d+)(?:/\d+)?\]", prompt_lower)
        if path_match:
            path_offset = int(path_match.group(1))

        bull_count = sum(prompt_lower.count(w) for w in bullish_cues)
        bear_count = sum(prompt_lower.count(w) for w in bearish_cues)

        if bull_count > bear_count:
            # Mostly positive with slight noise on high path number
            sentiment = "positive" if path_offset != 7 else "neutral"
            reason = "Strong forward earnings catalysts, positive revenue revision, and expanding retail call option interest."
        elif bear_count > bull_count:
            sentiment = "negative" if path_offset != 6 else "neutral"
            reason = "Guidance contraction, margin compression, and institutional downgrade pressure create downside risks."
        else:
            sentiment = "neutral"
            reason = "Balanced discussion weighing product pipeline upside against valuation multiples without clear directional bias."

        response_dict = {
            "reasoning_summary": reason,
            "sentiment": sentiment,
        }
        resp_text = json.dumps(response_dict)

        p_toks = len(prompt.split()) + 25
        c_toks = len(resp_text.split()) + 10
        return resp_text, p_toks, c_toks


def get_llm_provider(
    provider: Optional[str] = None,
    model_name: Optional[str] = None,
    temperature: float = 0.5,
    **kwargs,
) -> BaseLLMProvider:
    """
    Factory function instantiating the requested LLM provider.
    Priority: argument > environment variable > config file > mock fallback.
    """
    load_dotenv_if_available()
    provider_env = os.getenv("LLM_PROVIDER")
    model_env = os.getenv("LLM_MODEL")

    selected_provider = (provider or provider_env or "mock").lower()

    # Load configuration defaults if available
    try:
        cfg = load_config("model")
        t_cfg = cfg.get("teacher_llm", {})
    except Exception:
        t_cfg = {}

    if selected_provider == "openai":
        api_key = os.getenv("OPENAI_API_KEY")
        if api_key and not api_key.startswith("your_"):
            m = model_name or model_env or t_cfg.get("providers", {}).get("openai", {}).get("model", "gpt-4o-mini")
            return OpenAIProvider(model_name=m, temperature=temperature, **kwargs)
        logger.info("OPENAI_API_KEY not found. Falling back to MockProvider.")
        return MockProvider(temperature=temperature, **kwargs)

    elif selected_provider == "gemini":
        api_key = os.getenv("GEMINI_API_KEY")
        if api_key and not api_key.startswith("your_"):
            m = model_name or model_env or t_cfg.get("providers", {}).get("gemini", {}).get("model", "gemini-1.5-flash")
            return GeminiProvider(model_name=m, temperature=temperature, **kwargs)
        logger.info("GEMINI_API_KEY not found. Falling back to MockProvider.")
        return MockProvider(temperature=temperature, **kwargs)

    elif selected_provider == "anthropic":
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if api_key and not api_key.startswith("your_"):
            m = model_name or model_env or t_cfg.get("providers", {}).get("anthropic", {}).get("model", "claude-3-5-sonnet-20241022")
            return AnthropicProvider(model_name=m, temperature=temperature, **kwargs)
        logger.info("ANTHROPIC_API_KEY not found. Falling back to MockProvider.")
        return MockProvider(temperature=temperature, **kwargs)

    elif selected_provider == "local":
        m = model_name or model_env or t_cfg.get("providers", {}).get("local", {}).get("model", "llama3")
        base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        return LocalProvider(model_name=m, base_url=base_url, temperature=temperature, **kwargs)

    else:
        return MockProvider(temperature=temperature, **kwargs)
