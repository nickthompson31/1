"""
Depth estimation using Depth Anything V2.

Provides two quality tiers:
- Preview: Depth-Anything-V2-Small (fast, ~4s on CPU)
- Final: Depth-Anything-V2-Base (higher quality, ~8s on CPU)

Depth Anything V2 produces dramatically better facial feature detail
than MiDaS — visible nose, eye sockets, mouth, jawline out of the box.

Output is a normalized depth map where higher values = closer to camera.
"""

import numpy as np
import torch
from PIL import Image
from scipy import ndimage


class DepthEstimator:
    """Monocular depth estimation using Depth Anything V2."""

    MODELS = {
        "preview": "depth-anything/Depth-Anything-V2-Small-hf",
        "final": "depth-anything/Depth-Anything-V2-Base-hf",
    }

    def __init__(self):
        self._pipes: dict = {}
        self._device = -1  # CPU; use 0 for CUDA

    def _load_model(self, quality: str):
        """Lazy-load model on first use."""
        if quality in self._pipes:
            return

        from transformers import pipeline

        model_id = self.MODELS[quality]
        self._pipes[quality] = pipeline(
            task="depth-estimation",
            model=model_id,
            device=self._device,
        )

    def estimate(
        self,
        image: Image.Image,
        quality: str = "preview",
        target_size: int | None = None,
    ) -> np.ndarray:
        """
        Estimate depth from an image.

        Args:
            image: PIL Image (RGB)
            quality: "preview" (fast) or "final" (high quality)
            target_size: Optional max dimension to resize input before processing.

        Returns:
            Normalized depth map as numpy array (float32, 0-1 range).
            Higher values = closer to camera (foreground).
        """
        if quality not in self.MODELS:
            raise ValueError(f"Quality must be one of: {list(self.MODELS.keys())}")

        self._load_model(quality)

        # Optionally resize for speed
        img = image.copy()
        if target_size is not None:
            w, h = img.size
            scale = target_size / max(w, h)
            if scale < 1.0:
                img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)

        # Run inference
        result = self._pipes[quality](img)
        depth = np.array(result["depth"], dtype=np.float32)

        # Normalize to 0-1 range
        depth = depth - depth.min()
        max_val = depth.max()
        if max_val > 0:
            depth = depth / max_val

        # Depth Anything outputs "disparity" where closer=higher, which is
        # what we want (foreground = higher values)
        return depth.astype(np.float32)
