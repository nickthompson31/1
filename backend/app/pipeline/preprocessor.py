"""
Image preprocessing: background removal, cropping, normalization.

Prepares the input portrait for the depth estimation pipeline.
"""

import numpy as np
from PIL import Image, ImageFilter
from scipy import ndimage


class Preprocessor:
    """Prepares portrait images for the depth pipeline."""

    def process(
        self,
        image: Image.Image,
        remove_background: bool = True,
        auto_crop: bool = True,
        max_dimension: int = 1024,
        padding_percent: float = 0.05,
    ) -> tuple[Image.Image, np.ndarray | None]:
        """
        Preprocess a portrait image.

        Args:
            image: Input PIL Image
            remove_background: Whether to remove the background
            auto_crop: Whether to auto-crop to the subject
            max_dimension: Max pixel dimension (for processing speed)
            padding_percent: Padding around subject after crop (0-1)

        Returns:
            Tuple of (processed_image, background_mask_or_None)
        """
        img = image.copy()

        # Ensure RGB
        if img.mode != "RGB":
            img = img.convert("RGB")

        # Resize if too large
        w, h = img.size
        if max(w, h) > max_dimension:
            scale = max_dimension / max(w, h)
            img = img.resize(
                (int(w * scale), int(h * scale)), Image.LANCZOS
            )

        bg_mask = None

        if remove_background:
            img, bg_mask = self._remove_background(img)

        if auto_crop and bg_mask is not None:
            img, bg_mask = self._auto_crop(img, bg_mask, padding_percent)

        return img, bg_mask

    def _remove_background(
        self, image: Image.Image
    ) -> tuple[Image.Image, np.ndarray]:
        """
        Remove background using rembg.

        Returns the image (with white background) and a binary mask
        where 1.0 = foreground (subject), 0.0 = background.
        """
        try:
            from rembg import remove

            # Get image with alpha channel
            result = remove(image)

            # Extract alpha as mask
            if result.mode == "RGBA":
                alpha = np.array(result.split()[-1], dtype=np.float32) / 255.0
                # Threshold to clean up edges
                mask = (alpha > 0.5).astype(np.float32)

                # Smooth mask edges slightly
                mask = ndimage.gaussian_filter(mask, sigma=1)
                mask = np.clip(mask, 0, 1)

                # Create white-background version
                bg = Image.new("RGB", result.size, (255, 255, 255))
                bg.paste(result, mask=result.split()[-1])
                return bg, mask
            else:
                return result.convert("RGB"), np.ones(
                    (image.size[1], image.size[0]), dtype=np.float32
                )

        except ImportError:
            # rembg not available — return image as-is with full mask
            return image, np.ones(
                (image.size[1], image.size[0]), dtype=np.float32
            )

    def _auto_crop(
        self,
        image: Image.Image,
        mask: np.ndarray,
        padding_percent: float,
    ) -> tuple[Image.Image, np.ndarray]:
        """Crop image to the foreground subject with padding."""
        # Find bounding box of foreground
        rows = np.any(mask > 0.5, axis=1)
        cols = np.any(mask > 0.5, axis=0)

        if not rows.any() or not cols.any():
            return image, mask

        y_min, y_max = np.where(rows)[0][[0, -1]]
        x_min, x_max = np.where(cols)[0][[0, -1]]

        # Add padding
        h, w = mask.shape
        pad_x = int((x_max - x_min) * padding_percent)
        pad_y = int((y_max - y_min) * padding_percent)

        x_min = max(0, x_min - pad_x)
        x_max = min(w, x_max + pad_x)
        y_min = max(0, y_min - pad_y)
        y_max = min(h, y_max + pad_y)

        # Crop both image and mask
        cropped_img = image.crop((x_min, y_min, x_max, y_max))
        cropped_mask = mask[y_min:y_max, x_min:x_max]

        return cropped_img, cropped_mask
