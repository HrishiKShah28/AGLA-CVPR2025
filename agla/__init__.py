"""
AGLA: Assembly of Global and Local Attention for Mitigating Object Hallucinations in LVLMs
CVPR 2025
"""

from agla.ipm import ImagePromptMatcher
from agla.decoder import assemble_logits, apply_adaptive_plausibility_constraint, sample_next_token
from agla.models import BaseLVLM, HuggingFaceLVLM, SimulatedLVLM
from agla.evaluator import POPEEvaluator, compute_pope_metrics
from agla.utils import load_image, generate_heatmap_overlay, mask_image_with_relevance

__version__ = "1.0.0"
__all__ = [
    "ImagePromptMatcher",
    "assemble_logits",
    "apply_adaptive_plausibility_constraint",
    "sample_next_token",
    "BaseLVLM",
    "HuggingFaceLVLM",
    "SimulatedLVLM",
    "POPEEvaluator",
    "compute_pope_metrics",
    "load_image",
    "generate_heatmap_overlay",
    "mask_image_with_relevance"
]
