"""
Face-aware depth refinement — the "anti-creepy" engine.

This is the core innovation. Instead of treating the entire depth map uniformly,
we apply region-specific processing to each facial feature:

- Eyes: Prevent hollow/sunken appearance (the #1 cause of creepiness)
- Nose: Controlled projection with smooth gradients
- Mouth: Minimal depth variation to avoid grimace
- Hair: Simplified texture (individual strands look like worms in relief)
- Skin: Preserve likeness while removing noise

Each region gets its own depth treatment, smoothing, and blending.
"""

import numpy as np
from scipy import ndimage
from dataclasses import dataclass

from .face import FaceData


@dataclass
class RefinementParams:
    """User-adjustable parameters for depth refinement."""

    # Overall depth intensity (0.0 = flat, 1.0 = maximum relief)
    depth_intensity: float = 0.7

    # Eye depth control (0.0 = flat eyes, 1.0 = deep eye sockets)
    # Lower values prevent the hollow/dead eye look
    eye_depth: float = 0.3

    # Nose projection (0.0 = flat, 1.0 = maximum projection)
    nose_projection: float = 0.8

    # Mouth depth variation (0.0 = flat lips, 1.0 = deep lip detail)
    mouth_depth: float = 0.4

    # Forehead smoothing (higher = smoother, less wrinkle detail)
    forehead_smoothing: float = 0.6

    # Cheek smoothing
    cheek_smoothing: float = 0.5

    # Hair simplification (higher = more simplified/smooth hair)
    hair_simplification: float = 0.7

    # Global smoothing radius (pixels)
    smoothing_sigma: float = 2.0

    # Detail preservation (0.0 = smooth, 1.0 = max detail)
    detail_level: float = 0.6

    # Edge sharpness for face boundary
    edge_sharpness: float = 0.5


class DepthRefiner:
    """Applies face-aware depth refinement to raw depth maps."""

    def refine(
        self,
        depth_map: np.ndarray,
        faces: list[FaceData],
        params: RefinementParams | None = None,
    ) -> np.ndarray:
        """
        Apply face-aware refinement to a raw depth map.

        Args:
            depth_map: Raw normalized depth map (0-1, float32)
            faces: Detected face data with region masks
            params: User-adjustable refinement parameters

        Returns:
            Refined depth map (0-1, float32)
        """
        if params is None:
            params = RefinementParams()

        refined = depth_map.copy()

        if not faces:
            # No faces detected — apply basic smoothing only
            refined = self._apply_global_smoothing(refined, params)
            return refined

        for face in faces:
            refined = self._refine_face(refined, face, params)

        # Apply global smoothing to non-face areas
        face_mask = np.zeros_like(refined)
        for face in faces:
            face_mask = np.maximum(face_mask, face.region_masks.combined)

        # Smooth non-face areas more aggressively
        bg_smoothed = ndimage.gaussian_filter(refined, sigma=params.smoothing_sigma * 3)
        bg_weight = 1.0 - face_mask
        refined = refined * face_mask + bg_smoothed * bg_weight

        return refined

    def _refine_face(
        self,
        depth: np.ndarray,
        face: FaceData,
        params: RefinementParams,
    ) -> np.ndarray:
        """Apply region-specific depth treatment to a single face."""
        masks = face.region_masks
        result = depth.copy()

        # --- EYES: The most critical region ---
        # Problem: Raw depth often makes eye sockets too deep → "dead eyes"
        # Solution: Clamp depth range within eye region, reduce variation
        result = self._process_eyes(result, masks, params)

        # --- NOSE: Controlled projection ---
        # Problem: Nose tip can spike too high, nostrils can create weird holes
        # Solution: Smooth gradient from bridge to tip, controlled max height
        result = self._process_nose(result, masks, params)

        # --- MOUTH: Minimize depth variation ---
        # Problem: Lip depth variations create grimace/snarl appearance
        # Solution: Flatten lip area, gentle transition to surrounding face
        result = self._process_mouth(result, masks, params)

        # --- FOREHEAD & CHEEKS: Smooth broad surfaces ---
        result = self._process_broad_surfaces(result, masks, params)

        # --- BLEND regions smoothly ---
        result = self._blend_regions(result, depth, masks, params)

        return result

    def _process_eyes(
        self,
        depth: np.ndarray,
        masks,
        params: RefinementParams,
    ) -> np.ndarray:
        """
        Process eye regions to prevent hollow/dead eye appearance.

        Key insight: In bas-relief carving, eyes should have SUBTLE depth,
        not deep sockets. The depth range should be compressed significantly.
        """
        result = depth.copy()

        for eye_mask in [masks.left_eye, masks.right_eye]:
            if eye_mask.sum() == 0:
                continue

            eye_region = depth[eye_mask > 0.5]
            if len(eye_region) == 0:
                continue

            # Get the median depth of the eye region as our target
            median_depth = np.median(eye_region)
            face_median = np.median(depth[masks.face_outline > 0.5])

            # Compress eye depth range toward the face surface
            # Lower eye_depth = flatter eyes = less creepy
            compression = 1.0 - params.eye_depth

            # Pull eye depth toward face surface level
            target = face_median * 0.95  # Slightly recessed from face
            eye_values = depth.copy()
            eye_adjusted = target + (eye_values - target) * (1.0 - compression * 0.7)

            # Apply with smooth blending via the mask
            blurred_mask = ndimage.gaussian_filter(eye_mask, sigma=5)
            result = result * (1 - blurred_mask) + eye_adjusted * blurred_mask

        return result

    def _process_nose(
        self,
        depth: np.ndarray,
        masks,
        params: RefinementParams,
    ) -> np.ndarray:
        """
        Process nose region for controlled projection.

        The nose should be the highest point but with smooth gradients.
        Nostrils should NOT create deep holes.
        """
        result = depth.copy()
        nose_mask = masks.nose

        if nose_mask.sum() == 0:
            return result

        # Smooth the nose region to create clean gradients
        nose_smoothed = ndimage.gaussian_filter(result, sigma=3)

        # Scale nose projection
        face_depth = np.median(depth[masks.face_outline > 0.5])
        nose_region = nose_smoothed[nose_mask > 0.5]
        if len(nose_region) == 0:
            return result

        # Normalize nose depth relative to face
        nose_adjusted = result.copy()
        nose_delta = nose_adjusted - face_depth
        nose_adjusted = face_depth + nose_delta * params.nose_projection

        # Smooth the nose area
        nose_adjusted = ndimage.gaussian_filter(nose_adjusted, sigma=2)

        # Apply with soft mask
        blurred_mask = ndimage.gaussian_filter(nose_mask, sigma=8)
        result = result * (1 - blurred_mask) + nose_adjusted * blurred_mask

        return result

    def _process_mouth(
        self,
        depth: np.ndarray,
        masks,
        params: RefinementParams,
    ) -> np.ndarray:
        """
        Process mouth region to prevent grimace appearance.

        Lips should have very subtle depth variation in bas-relief.
        The lip line should be a gentle crease, not a deep gash.
        """
        result = depth.copy()
        mouth_mask = masks.mouth

        if mouth_mask.sum() == 0:
            return result

        # Get surrounding face depth as reference
        face_depth = np.median(depth[masks.face_outline > 0.5])

        # Compress mouth depth variation
        mouth_adjusted = result.copy()
        mouth_delta = mouth_adjusted - face_depth
        mouth_adjusted = face_depth + mouth_delta * params.mouth_depth

        # Extra smoothing on mouth region
        mouth_adjusted = ndimage.gaussian_filter(mouth_adjusted, sigma=3)

        # Apply with soft blending
        blurred_mask = ndimage.gaussian_filter(mouth_mask, sigma=6)
        result = result * (1 - blurred_mask) + mouth_adjusted * blurred_mask

        return result

    def _process_broad_surfaces(
        self,
        depth: np.ndarray,
        masks,
        params: RefinementParams,
    ) -> np.ndarray:
        """Smooth forehead and cheek areas for clean carving surfaces."""
        result = depth.copy()

        # Forehead: smooth to reduce wrinkle noise
        if masks.forehead.sum() > 0:
            sigma = 3 + params.forehead_smoothing * 7  # 3-10 pixel sigma
            forehead_smooth = ndimage.gaussian_filter(result, sigma=sigma)
            blurred_mask = ndimage.gaussian_filter(masks.forehead, sigma=5)
            result = result * (1 - blurred_mask) + forehead_smooth * blurred_mask

        # Cheeks
        for cheek_mask in [masks.left_cheek, masks.right_cheek]:
            if cheek_mask.sum() > 0:
                sigma = 2 + params.cheek_smoothing * 5
                cheek_smooth = ndimage.gaussian_filter(result, sigma=sigma)
                blurred_mask = ndimage.gaussian_filter(cheek_mask, sigma=5)
                result = result * (1 - blurred_mask) + cheek_smooth * blurred_mask

        return result

    def _blend_regions(
        self,
        refined: np.ndarray,
        original: np.ndarray,
        masks,
        params: RefinementParams,
    ) -> np.ndarray:
        """
        Blend refined depth with original to preserve detail.

        The detail_level parameter controls how much of the original
        high-frequency detail is preserved vs. the smoothed version.
        """
        # Extract high-frequency detail from original
        smooth_original = ndimage.gaussian_filter(original, sigma=params.smoothing_sigma)
        detail = original - smooth_original

        # Add back detail scaled by detail_level (only in face area)
        face_mask = ndimage.gaussian_filter(masks.combined, sigma=3)
        detail_contribution = detail * params.detail_level * face_mask

        result = refined + detail_contribution

        # Final light smoothing to blend everything
        result = ndimage.gaussian_filter(result, sigma=params.smoothing_sigma * 0.5)

        # Re-normalize to 0-1
        result = result - result.min()
        max_val = result.max()
        if max_val > 0:
            result = result / max_val

        return result.astype(np.float32)

    def _apply_global_smoothing(
        self, depth: np.ndarray, params: RefinementParams
    ) -> np.ndarray:
        """Apply basic smoothing when no face is detected."""
        smoothed = ndimage.gaussian_filter(depth, sigma=params.smoothing_sigma)

        # Blend original detail back in
        detail = depth - ndimage.gaussian_filter(depth, sigma=params.smoothing_sigma * 2)
        result = smoothed + detail * params.detail_level

        result = result - result.min()
        max_val = result.max()
        if max_val > 0:
            result = result / max_val

        return result.astype(np.float32)
