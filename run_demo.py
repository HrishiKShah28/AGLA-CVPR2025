"""
Interactive Demo Script: Baseline vs AGLA
Usage:
    python run_demo.py --image data/sample_images/skateboarder.jpg --query "Describe the objects in this image."
    python run_demo.py --image data/sample_images/skateboarder.jpg --query "Is there a car in the image?"
"""

import argparse
import os
import sys
from agla.ipm import ImagePromptMatcher
from agla.models import SimulatedLVLM, HuggingFaceLVLM
from agla.utils import load_image


def main():
    parser = argparse.ArgumentParser(description="AGLA Hallucination Mitigation Demo")
    parser.add_argument("--image", type=str, default="data/sample_images/skateboarder.jpg", help="Path to input image")
    parser.add_argument("--query", type=str, default="Describe the objects in this image.", help="Text prompt / question")
    parser.add_argument("--alpha", type=float, default=1.0, help="Weight of local logits (default: 1.0)")
    parser.add_argument("--beta", type=float, default=0.1, help="Adaptive plausibility constraint threshold (default: 0.1)")
    parser.add_argument("--use_real_model", action="store_true", help="Load real HuggingFace LVLM (requires GPU/RAM)")
    args = parser.parse_args()

    print("=" * 70)
    print("  CVPR 2025 AGLA: Mitigating Object Hallucinations in LVLMs")
    print("=" * 70)
    print(f"Image Path : {args.image}")
    print(f"Query      : {args.query}")
    print(f"Params     : alpha={args.alpha}, beta={args.beta}\n")

    image = load_image(args.image)

    # 1. Image-Prompt Matching (IPM)
    print(">>> Step 1: Running Image-Prompt Matching (IPM)...")
    matcher = ImagePromptMatcher()
    ipm_res = matcher.process(image, args.query)
    print(f"    - Image-Prompt Similarity sim(v, t) : {ipm_res['similarity_score']:.4f}")
    print(f"    - Adaptive Masking Ratio (sim / 2)  : {ipm_res['mask_ratio']:.4f} ({ipm_res['mask_ratio']*100:.1f}%)")

    # Save augmented image & heatmap
    os.makedirs("outputs", exist_ok=True)
    ipm_res["heatmap_image"].save("outputs/gradcam_heatmap.jpg")
    ipm_res["augmented_image"].save("outputs/augmented_masked.jpg")
    print("    - Saved attention heatmap to 'outputs/gradcam_heatmap.jpg'")
    print("    - Saved augmented view to 'outputs/augmented_masked.jpg'")

    # 2. LVLM Model Setup
    print("\n>>> Step 2: Preparing LVLM...")
    if args.use_real_model:
        lvlm = HuggingFaceLVLM(model_id="HuggingFaceTB/SmolVLM-500M-Instruct")
    else:
        lvlm = SimulatedLVLM()

    # 3. Baseline Decoding
    print("\n>>> Step 3: Running Baseline LVLM Decoding (Global Original View only)...")
    base_res = lvlm.generate_baseline(image=image, prompt=args.query)
    print("----------------------------------------------------------------------")
    print("[BASELINE OUTPUT]:")
    print(base_res["response"])
    print("----------------------------------------------------------------------")

    # 4. AGLA Calibrated Decoding
    print("\n>>> Step 4: Running AGLA Calibrated Decoding (Logit Assembly + Plausibility Filtering)...")
    agla_res = lvlm.generate_agla(
        orig_image=image,
        aug_image=ipm_res["augmented_image"],
        prompt=args.query,
        alpha=args.alpha,
        beta=args.beta,
        apply_truncation=True
    )
    print("----------------------------------------------------------------------")
    print("[AGLA CALIBRATED OUTPUT]:")
    print(agla_res["response"])
    print("----------------------------------------------------------------------")

    print("\n" + "=" * 70)
    print("  COMPARISON SUMMARY:")
    print("=" * 70)
    print(f"Baseline Hallucination Status : {'HALLUCINATED OBJECTS DETECTED' if base_res.get('hallucinated') else 'Grounded'}")
    print(f"AGLA Hallucination Status     : {'HALLUCINATED' if agla_res.get('hallucinated') else 'GROUNDED & FACTUAL (Hallucinations Mitigated)'}")
    print("=" * 70)


if __name__ == "__main__":
    main()
