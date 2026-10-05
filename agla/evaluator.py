"""
POPE Evaluation Benchmark and Ablation Study Engine
Implements Section 4.1 & 4.3 of CVPR 2025 Paper:
- Polling-based Object Probing Evaluation (POPE)
- Subsets: Random, Popular, Adversarial
- Metrics: Accuracy, Precision, Recall, F1 Score
- Ablation Studies: w/o Truncation, w/o Adaptive Ratio, w/o Assembly
"""

import json
from typing import Dict, List, Any, Optional
import numpy as np


def compute_pope_metrics(predictions: List[str], ground_truths: List[str]) -> Dict[str, float]:
    """
    Computes Accuracy, Precision, Recall, and F1 Score for Yes/No POPE questions.
    """
    tp, fp, tn, fn = 0, 0, 0, 0
    for pred, gt in zip(predictions, ground_truths):
        p_yes = "yes" in pred.lower()
        g_yes = "yes" in gt.lower()

        if p_yes and g_yes:
            tp += 1
        elif p_yes and not g_yes:
            fp += 1
        elif not p_yes and not g_yes:
            tn += 1
        elif not p_yes and g_yes:
            fn += 1

    total = tp + fp + tn + fn
    acc = (tp + tn) / total if total > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "accuracy": round(acc * 100, 2),
        "precision": round(precision * 100, 2),
        "recall": round(recall * 100, 2),
        "f1": round(f1 * 100, 2),
        "tp": tp, "fp": fp, "tn": tn, "fn": fn, "total": total
    }


# Standard POPE questions matching CVPR 2025 experimental setup (50/50 Positive/Negative balance)
DEFAULT_POPE_DATASET = {
    "random": [
        {"image_name": "skateboarder.jpg", "question": "Is there a person in the image?", "ground_truth": "Yes"},
        {"image_name": "skateboarder.jpg", "question": "Is there a skateboard in the image?", "ground_truth": "Yes"},
        {"image_name": "skateboarder.jpg", "question": "Is there an elephant in the image?", "ground_truth": "No"},
        {"image_name": "skateboarder.jpg", "question": "Is there an airplane in the image?", "ground_truth": "No"},
        {"image_name": "tennis.jpg", "question": "Is there a person in the image?", "ground_truth": "Yes"},
        {"image_name": "tennis.jpg", "question": "Is there a tennis racket in the image?", "ground_truth": "Yes"},
        {"image_name": "tennis.jpg", "question": "Is there a giraffe in the image?", "ground_truth": "No"},
        {"image_name": "tennis.jpg", "question": "Is there a refrigerator in the image?", "ground_truth": "No"}
    ],
    "popular": [
        {"image_name": "skateboarder.jpg", "question": "Is there a person in the image?", "ground_truth": "Yes"},
        {"image_name": "skateboarder.jpg", "question": "Is there a skateboard in the image?", "ground_truth": "Yes"},
        {"image_name": "skateboarder.jpg", "question": "Is there a chair in the image?", "ground_truth": "No"},
        {"image_name": "skateboarder.jpg", "question": "Is there a dining table in the image?", "ground_truth": "No"},
        {"image_name": "tennis.jpg", "question": "Is there a sports ball in the image?", "ground_truth": "Yes"},
        {"image_name": "tennis.jpg", "question": "Is there a tennis racket in the image?", "ground_truth": "Yes"},
        {"image_name": "tennis.jpg", "question": "Is there a chair in the image?", "ground_truth": "No"},
        {"image_name": "tennis.jpg", "question": "Is there a bottle in the image?", "ground_truth": "No"}
    ],
    "adversarial": [
        {"image_name": "skateboarder.jpg", "question": "Is there a person in the image?", "ground_truth": "Yes"},
        {"image_name": "skateboarder.jpg", "question": "Is there a skateboard in the image?", "ground_truth": "Yes"},
        {"image_name": "skateboarder.jpg", "question": "Is there a car in the image?", "ground_truth": "No"},
        {"image_name": "skateboarder.jpg", "question": "Is there a bicycle in the image?", "ground_truth": "No"},
        {"image_name": "tennis.jpg", "question": "Is there a tennis racket in the image?", "ground_truth": "Yes"},
        {"image_name": "tennis.jpg", "question": "Is there a net in the image?", "ground_truth": "Yes"},
        {"image_name": "tennis.jpg", "question": "Is there a referee in the image?", "ground_truth": "No"},
        {"image_name": "tennis.jpg", "question": "Is there a scoreboard in the image?", "ground_truth": "No"}
    ]
}


class POPEEvaluator:
    """Runs evaluation benchmarks on POPE splits and ablations."""

    def __init__(self, pope_data: Optional[Dict[str, List[Dict[str, str]]]] = None):
        self.pope_data = pope_data or DEFAULT_POPE_DATASET

    def evaluate_model(
        self,
        matcher: Any,
        lvlm: Any,
        images_dict: Dict[str, Any],
        subset: str = "adversarial",
        mode: str = "agla",
        alpha: float = 1.0,
        beta: float = 0.1,
        apply_truncation: bool = True,
        fixed_mask_ratio: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Runs evaluation on a specific POPE subset.
        Modes: 'baseline', 'agla', 'ablation_no_assembly'
        """
        questions = self.pope_data.get(subset, [])
        preds = []
        gts = []
        details = []

        for item in questions:
            img = images_dict[item["image_name"]]
            q = item["question"]
            gt = item["ground_truth"]

            if mode == "baseline":
                out = lvlm.generate_baseline(image=img, prompt=q)
                pred_text = out["response"]
            elif mode == "agla":
                ipm_res = matcher.process(img, q, fixed_mask_ratio=fixed_mask_ratio)
                out = lvlm.generate_agla(
                    orig_image=img,
                    aug_image=ipm_res["augmented_image"],
                    prompt=q,
                    alpha=alpha,
                    beta=beta,
                    apply_truncation=apply_truncation
                )
                pred_text = out["response"]
            elif mode == "ablation_no_assembly":
                # Augmented image only (w/o assembly)
                ipm_res = matcher.process(img, q, fixed_mask_ratio=fixed_mask_ratio)
                out = lvlm.generate_baseline(image=ipm_res["augmented_image"], prompt=q)
                pred_text = out["response"]
            else:
                raise ValueError(f"Unknown mode: {mode}")

            preds.append(pred_text)
            gts.append(gt)

            # Check correctness
            p_yes = "yes" in pred_text.lower()
            g_yes = "yes" in gt.lower()
            is_correct = (p_yes == g_yes)

            details.append({
                "question": q,
                "ground_truth": gt,
                "prediction": pred_text,
                "correct": is_correct
            })

        metrics = compute_pope_metrics(preds, gts)
        metrics["subset"] = subset
        metrics["mode"] = mode
        metrics["details"] = details
        return metrics
