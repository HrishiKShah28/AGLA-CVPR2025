"""
AGLA Decoding and Logit Assembly Engine
Implements Section 3.2 of CVPR 2025 Paper:
"Mitigating Object Hallucinations in Large Vision-Language Models with Assembly of Global and Local Attention"
"""

from typing import Dict, List, Optional, Tuple, Any
import torch
import torch.nn.functional as F


def apply_adaptive_plausibility_constraint(
    orig_logits: torch.Tensor,
    assembled_logits: torch.Tensor,
    beta: float = 0.1
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Implements Equation (4) of the paper:
    V_token = { y_i in V : P_orig(y_i) >= beta * max_w P_orig(w) }
    P_AGLA(y_i) = 0 if y_i not in V_token (logits set to -inf)
    """
    orig_probs = F.softmax(orig_logits, dim=-1)
    max_orig_prob = torch.max(orig_probs, dim=-1, keepdim=True).values
    
    threshold = beta * max_orig_prob
    plausible_mask = orig_probs >= threshold
    
    filtered_logits = assembled_logits.clone()
    # Mask out non-plausible candidate tokens
    filtered_logits[~plausible_mask] = -float("inf")
    
    return filtered_logits, plausible_mask


def assemble_logits(
    orig_logits: torch.Tensor,
    aug_logits: torch.Tensor,
    alpha: float = 1.0,
    beta: float = 0.1,
    apply_truncation: bool = True
) -> Dict[str, Any]:
    """
    Assembles logits from original image (global) and augmented image (local).
    Formula: L_AGLA = L_orig + alpha * L_aug
    followed by adaptive plausibility filtering with parameter beta.
    """
    # Eq (3): L_AGLA = L_orig + alpha * L_aug
    raw_assembled_logits = orig_logits + alpha * aug_logits

    if apply_truncation and beta > 0.0:
        calibrated_logits, plausible_mask = apply_adaptive_plausibility_constraint(
            orig_logits=orig_logits,
            assembled_logits=raw_assembled_logits,
            beta=beta
        )
    else:
        calibrated_logits = raw_assembled_logits
        plausible_mask = torch.ones_like(raw_assembled_logits, dtype=torch.bool)

    calibrated_probs = F.softmax(calibrated_logits, dim=-1)
    orig_probs = F.softmax(orig_logits, dim=-1)
    aug_probs = F.softmax(aug_logits, dim=-1)

    return {
        "calibrated_logits": calibrated_logits,
        "calibrated_probs": calibrated_probs,
        "raw_assembled_logits": raw_assembled_logits,
        "orig_probs": orig_probs,
        "aug_probs": aug_probs,
        "plausible_mask": plausible_mask
    }


def sample_next_token(
    logits: torch.Tensor,
    strategy: str = "greedy",
    temperature: float = 1.0,
    top_p: float = 0.7,
    top_k: int = 50
) -> int:
    """
    Selects the next token according to the specified decoding strategy.
    Supports: 'greedy', 'top_p', 'top_k', 'temperature', 'top_p_temp' (Table 6 in Paper).
    """
    if strategy == "greedy":
        return int(torch.argmax(logits, dim=-1).item())

    # Apply temperature
    if temperature > 0.0 and temperature != 1.0:
        logits = logits / temperature

    probs = F.softmax(logits, dim=-1)

    if "top_k" in strategy and top_k > 0:
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = -float("inf")
        probs = F.softmax(logits, dim=-1)

    if "top_p" in strategy and 0.0 < top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probs > top_p
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        logits[0, indices_to_remove] = -float("inf")
        probs = F.softmax(logits, dim=-1)

    # Multinomial sampling
    next_token = int(torch.multinomial(probs, num_samples=1).item())
    return next_token
