"""
Prompt Template Manager and In-Context Demonstration Formatter.

Loads versioned prompts from configs/prompts/sentiment_prompt.yaml, formats
the paper's 6 demonstrations (2 positive, 2 neutral, 2 negative) with CoT
financial reasoning, and supports ablation variants (e.g. shuffling demo order).
"""

import random
from typing import Dict, Any, List, Optional
from src.utils.config import load_prompt_config
from src.utils.logging import get_logger

logger = get_logger(__name__)


class PromptManager:
    """Manages prompt rendering, demonstration formatting, and ablations."""

    def __init__(self, prompt_config_name: str = "sentiment_prompt"):
        self.config = load_prompt_config(prompt_config_name)
        self.version = self.config.get("version", "1.0.0")
        self.system_prompt = self.config.get("system_prompt", "").strip()
        self.instruction = self.config.get("instruction", "").strip()
        self.demonstrations = self.config.get("demonstrations", [])
        self.template = self.config.get("template", "")

    def format_demonstrations(
        self,
        include_cot: bool = True,
        shuffle_seed: Optional[int] = None,
    ) -> str:
        """
        Format the 6 reference demonstrations.
        Supports shuffling order for demonstration order ablation experiments.
        """
        demos = list(self.demonstrations)
        if shuffle_seed is not None:
            rng = random.Random(shuffle_seed)
            rng.shuffle(demos)

        formatted_blocks = []
        for i, demo in enumerate(demos, 1):
            ticker = demo.get("ticker", "UNKNOWN")
            text = demo.get("text", "")
            sentiment = demo.get("sentiment", "neutral")
            reasoning = demo.get("reasoning_summary", "")

            if include_cot:
                block = (
                    f"Example {i}:\n"
                    f"Ticker: {ticker}\n"
                    f"Post: \"\"\"{text}\"\"\"\n"
                    f"Output:\n"
                    f'{{\n  "reasoning_summary": "{reasoning}",\n  "sentiment": "{sentiment}"\n}}'
                )
            else:
                block = (
                    f"Example {i}:\n"
                    f"Ticker: {ticker}\n"
                    f"Post: \"\"\"{text}\"\"\"\n"
                    f"Output:\n"
                    f'{{\n  "sentiment": "{sentiment}"\n}}'
                )
            formatted_blocks.append(block)

        return "\n\n".join(formatted_blocks)

    def build_prompt(
        self,
        post_text: str,
        ticker: str = "the mentioned company",
        include_cot: bool = True,
        shuffle_seed: Optional[int] = None,
    ) -> str:
        """
        Build the complete evaluation prompt for an input post.
        """
        demos_str = self.format_demonstrations(
            include_cot=include_cot, shuffle_seed=shuffle_seed
        )

        prompt = self.template.format(
            system_prompt=self.system_prompt,
            instruction=self.instruction,
            demonstrations_formatted=demos_str,
            ticker=ticker,
            post_text=post_text,
        )
        return prompt.strip()


def get_default_prompt_manager() -> PromptManager:
    """Convenience factory returning the default 6-shot prompt manager."""
    return PromptManager("sentiment_prompt")
