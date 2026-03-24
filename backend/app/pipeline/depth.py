"""
Depth estimation using MiDaS.

Provides two quality tiers:
- Preview: DPT-Small (fast, ~5-10s on CPU)
- Final: DPT-Large (high quality, ~30-60s on CPU)

Output is a normalized depth map where higher values = closer to camera.
"""

import numpy as np
import torch
from PIL import Image


class DepthEstimator:
    """Monocular depth estimation using MiDaS v3.1."""

    # Model configs: name -> (model_type, transform_type)
    MODELS = {
        "preview": ("DPT_Small", "small"),
        "final": ("DPT_Large", "dpt_large"),
    }

    def __init__(self):
        self._models: dict = {}
        self._transforms: dict = {}
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _load_model(self, quality: str):
        """Lazy-load a MiDaS model on first use."""
        if quality in self._models:
            return

        model_type = self.MODELS[quality][0]

        model = torch.hub.load("intel-isl/MiDaS", model_type, trust_repo=True)
        model.to(self._device)
        model.eval()
        self._models[quality] = model

        midas_transforms = torch.hub.load(
            "intel-isl/MiDaS", "transforms", trust_repo=True
        )
        if quality == "preview":
            self._transforms[quality] = midas_transforms.small_transform
        else:
            self._transforms[quality] = midas_transforms.dpt_transform

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
                         Smaller = faster. None = use model's default.

        Returns:
            Normalized depth map as numpy array (float32, 0-1 range).
            Higher values = closer to camera (foreground).
            Same aspect ratio as input, but resolution may differ.
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

        # Convert to numpy RGB
        img_np = np.array(img)

        # Apply MiDaS transform
        transform = self._transforms[quality]
        input_batch = transform(img_np).to(self._device)

        # Run inference
        with torch.no_grad():
            prediction = self._models[quality](input_batch)

            # Interpolate to original image size
            prediction = torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=img_np.shape[:2],
                mode="bicubic",
                align_corners=False,
            ).squeeze()

        depth = prediction.cpu().numpy()

        # Normalize to 0-1 range (invert so closer = higher)
        depth = depth - depth.min()
        max_val = depth.max()
        if max_val > 0:
            depth = depth / max_val

        return depth.astype(np.float32)
