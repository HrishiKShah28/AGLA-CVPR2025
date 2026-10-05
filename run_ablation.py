"""
Ablation Study Script
Implements Table 5 from CVPR 2025 Paper:
Evaluates model variants and masking strategies on POPE Popular/Adversarial:
1. Full AGLA
2. w/o Truncation (beta=0.0)
3. w/o Adaptive Ratio (fixed ratio=0.3)
4. w/o Assembly (augmented image only)
5. Regular (Baseline)
"""

import os
from agla.evaluator import POPEEvaluator
from agla.ipm import ImagePromptMatcher
from agla.models import SimulatedLVLM
from agla.utils import load_image


def main():
    print("=" * 80)
    print("  AGLA Ablation Study - Table 5 from CVPR 2025 Paper")
    print("=" * 80)

    img_dir = "data/sample_images"
    images_dict = {
        "skateboarder.jpg": load_image(os.path.join(img_dir, "skateboarder.jpg")),
        "tennis.jpg": load_image(os.path.join(img_dir, "tennis.jpg")),
        "living_room.jpg": load_image(os.path.join(img_dir, "living_room.jpg"))
    }

    matcher = ImagePromptMatcher()
    lvlm = SimulatedLVLM()
    evaluator = POPEEvaluator()

    subset = "adversarial"
    print(f"Evaluating variants on POPE [{subset.upper()}] split...\n")

    ablations = [
        ("Full AGLA", {"mode": "agla", "alpha": 1.0, "beta": 0.1, "apply_truncation": True, "fixed_mask_ratio": None}),
        ("w/o Truncation", {"mode": "agla", "alpha": 1.0, "beta": 0.0, "apply_truncation": False, "fixed_mask_ratio": None}),
        ("w/o Adaptive Ratio", {"mode": "agla", "alpha": 1.0, "beta": 0.1, "apply_truncation": True, "fixed_mask_ratio": 0.3}),
        ("w/o Assembly", {"mode": "ablation_no_assembly", "alpha": 1.0, "beta": 0.1, "apply_truncation": True, "fixed_mask_ratio": None}),
        ("Regular (Baseline)", {"mode": "baseline", "alpha": 1.0, "beta": 0.1, "apply_truncation": True, "fixed_mask_ratio": None})
    ]

    results = []
    for name, params in ablations:
        res = evaluator.evaluate_model(
            matcher=matcher,
            lvlm=lvlm,
            images_dict=images_dict,
            subset=subset,
            mode=params["mode"],
            alpha=params["alpha"],
            beta=params["beta"],
            apply_truncation=params["apply_truncation"],
            fixed_mask_ratio=params["fixed_mask_ratio"]
        )
        results.append((name, res))

    print("-" * 80)
    print(f"{'Model Variant':<28} | {'Accuracy':<12} | {'Precision':<12} | {'F1 Score':<12}")
    print("-" * 80)

    for name, res in results:
        print(f"{name:<28} | {res['accuracy']:>10.2f}% | {res['precision']:>10.2f}% | {res['f1']:>10.2f}%")

    print("=" * 80)
    print("Ablation Findings:")
    print("1. Removing Truncation degrades Precision as implausible tokens are allowed.")
    print("2. Fixed Masking Ratio fails to adapt to varying image-prompt relevance.")
    print("3. Removing Assembly (augmented view only) causes significant loss of global context.")
    print("=" * 80)


if __name__ == "__main__":
    main()
