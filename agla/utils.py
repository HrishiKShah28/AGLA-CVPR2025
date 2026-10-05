"""
AGLA Utility Functions: Image loading, heatmap generation, masking, and visualization.
"""

import io
from typing import Union, Tuple
import numpy as np
import torch
from PIL import Image
import cv2


def load_image(image_input: Union[str, Image.Image, bytes]) -> Image.Image:
    """Loads an image from a filepath, PIL Image, or raw bytes into RGB format."""
    if isinstance(image_input, Image.Image):
        return image_input.convert("RGB")
    elif isinstance(image_input, bytes):
        return Image.open(io.BytesIO(image_input)).convert("RGB")
    elif isinstance(image_input, str):
        return Image.open(image_input).convert("RGB")
    else:
        raise ValueError(f"Unsupported image input type: {type(image_input)}")


def generate_heatmap_overlay(
    image: Image.Image,
    relevance_map: np.ndarray,
    alpha: float = 0.55,
    colormap: int = cv2.COLORMAP_JET
) -> Image.Image:
    """
    Overlays a GradCAM relevance heatmap onto an image.
    relevance_map: 2D numpy array [0, 1]
    """
    img_np = np.array(image.convert("RGB"))
    h, w = img_np.shape[:2]

    # Resize relevance map to original image resolution
    cam_resized = cv2.resize(relevance_map.astype(np.float32), (w, h))
    cam_normalized = np.clip((cam_resized - cam_resized.min()) / (cam_resized.max() - cam_resized.min() + 1e-8), 0, 1)

    heatmap = cv2.applyColorMap(np.uint8(255 * cam_normalized), colormap)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

    overlay = (alpha * heatmap + (1 - alpha) * img_np).astype(np.uint8)
    return Image.fromarray(overlay)


def mask_image_with_relevance(
    image: Image.Image,
    relevance_map: np.ndarray,
    mask_ratio: float,
    mode: str = "patch",
    fill_color: Tuple[int, int, int] = (0, 0, 0)
) -> Image.Image:
    """
    Implements the Paper's Adaptive Masking strategy:
    Masks out image regions with the lowest correlation scores.
    mask_ratio: fraction of the lowest-scoring regions to mask (between 0.0 and 0.9).
    """
    img_np = np.array(image.convert("RGB"))
    h, w = img_np.shape[:2]

    cam_resized = cv2.resize(relevance_map.astype(np.float32), (w, h))
    
    # Threshold for bottom `mask_ratio` fraction
    if mask_ratio <= 0.0:
        return image.copy()
    
    mask_ratio = min(0.9, max(0.0, mask_ratio))
    threshold = np.quantile(cam_resized, mask_ratio)
    
    # 1 for keep (high relevance), 0 for mask (low relevance)
    binary_mask = (cam_resized >= threshold).astype(np.float32)

    if mode == "soft":
        # Soft weighting by normalized relevance
        norm_cam = (cam_resized - cam_resized.min()) / (cam_resized.max() - cam_resized.min() + 1e-8)
        masked_np = (img_np * norm_cam[..., None]).astype(np.uint8)
    else:
        # Patch/hard masking (paper default)
        fill = np.zeros_like(img_np)
        for c in range(3):
            fill[..., c] = fill_color[c]
        masked_np = (img_np * binary_mask[..., None] + fill * (1.0 - binary_mask[..., None])).astype(np.uint8)

    return Image.fromarray(masked_np)
