"""
Student Model Architectures for Knowledge Distillation.

Addresses Section 11 & 12:
- Note on Charformer: Charformer (102M parameter character-level T5) is proprietary
  and unsupported in modern Hugging Face/PyTorch ecosystems.
- Modernized with configurable compact transformer encoders:
  * distilbert-base-uncased (default, 66M params)
  * microsoft/deberta-v3-small (44M params)
  * sentence-transformers/all-MiniLM-L6-v2 (33M params)
- Supports dual heads:
  * Regression Head (num_labels=1, scalar continuous sentiment prediction)
  * Classification Head (num_labels=3, discrete logits)
"""

from typing import Tuple, Optional
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoConfig

from src.utils.logging import get_logger

logger = get_logger(__name__)

CHARFORMER_MODERNIZATION_NOTE = """
[MODERNIZATION NOTE]: The original paper (Deng et al., WWW '23) trained a 102M parameter
Charformer (character-level T5) pre-trained on proprietary Google internal social media data.
Because Charformer is not maintained in Hugging Face and weights are not publicly released,
we adopt modern sub-word transformer encoders (DistilBERT / DeBERTa-v3) that provide equivalent
compact parameter scale and fast edge inference while faithfully reproducing the paper's
regression loss distillation methodology.
"""


def build_student_model(
    backbone_name: str = "distilbert-base-uncased",
    objective: str = "regression",
    device: Optional[torch.device] = None,
) -> Tuple[nn.Module, AutoTokenizer]:
    """
    Factory function instantiating student model and tokenizer.
    objective: 'regression' (num_labels=1) or 'classification' (num_labels=3).
    """
    device = device or torch.device("cpu")
    num_labels = 1 if objective == "regression" else 3

    logger.info(
        f"Building student model backbone '{backbone_name}' with {objective.upper()} head (num_labels={num_labels})..."
    )

    tokenizer = AutoTokenizer.from_pretrained(backbone_name)

    # For sequence classification / regression, AutoModelForSequenceClassification appends a linear projection head
    model = AutoModelForSequenceClassification.from_pretrained(
        backbone_name,
        num_labels=num_labels,
    )
    model.to(device)

    return model, tokenizer
