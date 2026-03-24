"""
Face-aware depth refinement — the "anti-creepy" engine.

KEY INSIGHT from testing: The raw MiDaS depth map has the face at 0.9-1.0
with VERY subtle internal variation (~0.05 range). The original approach
of smoothing regions destroyed this detail.

NEW APPROACH: ENHANCE depth contrast within the face, not reduce it.
- Locally stretch the depth range within the face to reveal features
- Boost nose projection above the face plane
- Add controlled eye socket depth (subtle, not hollow)
- Preserve mouth definition but limit it
- Only smooth specific artifacts, not broad regions
"""

import numpy as np
from scipy import ndimage
from dataclasses import dataclass

from .face import FaceData


@dataclass
class RefinementParams:
    """User-adjustable parameters for depth refinement."""

    # Overall feature enhancement within face (0=flat, 1=maximum feature contrast)
    feature_strength: float = 0.8

    # Eye depth control: how much eyes recess below face surface
    # 0.0 = flush with face (flat), 0.5 = natural subtle recess, 1.0 = deep sockets
    eye_depth: float = 0.4

    # Nose projection boost above face surface
    # 0.0 = no boost, 1.0 = strong projection
    nose_projection: float = 0.85

    # Mouth/lip definition (0=flat, 1=deep lip line)
    mouth_depth: float = 0.5

    # Forehead curvature emphasis
    forehead_roundness: float = 0.6

    # Cheek volume emphasis
    cheek_volume: float = 0.6

    # Jaw/chin definition
    jaw_definition: float = 0.6

    # Global smoothing (post-enhancement cleanup)
    # Low values = more detail, high = smoother
    smoothing: float = 0.3

    # Detail preservation from original depth map (0=fully enhanced, 1=mostly original)
    detail_level: float = 0.5

    # Background smoothing (doesn't affect face)
    background_smoothing: float = 3.0


class DepthRefiner:
    """Enhances facial depth features for natural-looking bas-relief."""

    def refine(
        self,
        depth_map: np.ndarray,
        faces: list[FaceData],
        params: RefinementParams | None = None,
    ) -> np.ndarray:
        if params is None:
            params = RefinementParams()

        refined = depth_map.copy()

        if not faces:
            # No faces — just light smoothing
            refined = ndimage.gaussian_filter(refined, sigma=1.0)
            return self._normalize(refined)

        for face in faces:
            refined = self._enhance_face(refined, face, params)

        # Smooth non-face background areas
        face_mask = np.zeros_like(refined)
        for face in faces:
            face_mask = np.maximum(face_mask, face.region_masks.combined)

        # Feather the face mask edge for smooth blending
        face_mask_soft = ndimage.gaussian_filter(face_mask, sigma=8)
        face_mask_soft = np.clip(face_mask_soft, 0, 1)

        bg_smoothed = ndimage.gaussian_filter(refined, sigma=params.background_smoothing)
        refined = refined * face_mask_soft + bg_smoothed * (1 - face_mask_soft)

        # CRITICAL: Redistribute face depth range
        # The face typically occupies 0.8-1.0 of the depth range.
        # For bas-relief, we need the face internal features to span
        # a much wider range so the relief curve has contrast to work with.
        refined = self._redistribute_face_depth(refined, faces, params)

        return self._normalize(refined)

    def _enhance_face(
        self,
        depth: np.ndarray,
        face: FaceData,
        params: RefinementParams,
    ) -> np.ndarray:
        """Enhance facial feature contrast within the face region."""
        masks = face.region_masks
        result = depth.copy()

        face_pixels = depth[masks.face_outline > 0.5]
        if len(face_pixels) == 0:
            return result

        face_median = np.median(face_pixels)
        face_min = np.percentile(face_pixels, 5)
        face_max = np.percentile(face_pixels, 95)
        face_range = face_max - face_min

        # If face is very flat (common with MiDaS on close-up portraits),
        # we need to EXPAND the contrast significantly
        # Local contrast enhancement — moderate since redistribution
        # does the main range stretching afterward
        if face_range < 0.1:
            expansion_factor = 0.12 / max(face_range, 0.001) * params.feature_strength
        else:
            expansion_factor = params.feature_strength * 2.0

        # Step 1: Local contrast enhancement within face
        result = self._local_contrast_enhance(result, masks.face_outline, expansion_factor)

        # Step 2: Enhance specific features
        result = self._enhance_nose(result, masks, face_median, params)
        result = self._enhance_eyes(result, masks, face_median, params)
        result = self._enhance_mouth(result, masks, face_median, params)
        result = self._enhance_cheeks_jaw(result, masks, face_median, params)
        result = self._enhance_forehead(result, masks, face_median, params)

        # Step 3: Blend enhanced version with original for detail preservation
        blend_mask = ndimage.gaussian_filter(masks.combined, sigma=5)
        blend_mask = np.clip(blend_mask, 0, 1)

        # Mix: params.detail_level controls how much of original texture to keep
        original_detail = depth - ndimage.gaussian_filter(depth, sigma=1.5)
        result = result + original_detail * params.detail_level * blend_mask

        # Step 4: Blend smoothing within face to clean up any remaining mask artifacts
        if params.smoothing > 0:
            smooth_sigma = 1.0 + params.smoothing * 2.0  # 1.0-1.6 sigma range
            face_smoothed = ndimage.gaussian_filter(result, sigma=smooth_sigma)
            # Blend 40% smoothed to soften any remaining edges
            result = result * (1 - blend_mask * 0.4) + face_smoothed * (blend_mask * 0.4)

        return result

    def _local_contrast_enhance(
        self,
        depth: np.ndarray,
        face_mask: np.ndarray,
        expansion_factor: float,
    ) -> np.ndarray:
        """
        Expand depth contrast within the face region using CLAHE-like approach.

        The face typically occupies a narrow depth band (e.g., 0.90-0.95).
        This stretches that band to use more of the available range,
        revealing the subtle nose/eye/mouth depth variations.
        """
        result = depth.copy()
        face_region = face_mask > 0.5

        if not face_region.any():
            return result

        face_values = depth[face_region]
        face_median = np.median(face_values)

        # Stretch depth variations around the face median
        # This amplifies the subtle nose bump, eye recesses, etc.
        deviation = depth - face_median
        enhanced = face_median + deviation * expansion_factor

        # Apply only within face region with heavily softened blending
        soft_mask = ndimage.gaussian_filter(face_mask, sigma=10)
        soft_mask = np.clip(soft_mask, 0, 1)
        result = result * (1 - soft_mask) + enhanced * soft_mask

        return result

    def _enhance_nose(
        self,
        depth: np.ndarray,
        masks,
        face_median: float,
        params: RefinementParams,
    ) -> np.ndarray:
        """Boost nose projection above face surface."""
        result = depth.copy()
        nose_mask = masks.nose

        if nose_mask.sum() < 10:
            return result

        # Create a smooth nose projection profile
        # The nose should be the highest point on the face
        nose_region = depth[nose_mask > 0.5]
        if len(nose_region) == 0:
            return result

        # Boost nose projection with a very smooth, wide gaussian
        # so the enhancement blends naturally into cheeks/forehead
        # Reduced from 0.12 — redistribution now amplifies these
        boost_amount = 0.06 * params.nose_projection

        # Wide gaussian blur on mask creates a smooth natural mound shape
        # instead of following the polygon outline
        nose_smooth = ndimage.gaussian_filter(nose_mask, sigma=14)
        nose_smooth = nose_smooth / max(nose_smooth.max(), 1e-6)

        # Concentrate boost more at nose center by squaring the mask
        nose_focused = nose_smooth ** 1.5
        nose_focused = nose_focused / max(nose_focused.max(), 1e-6)

        boost = nose_focused * boost_amount
        result = result + boost

        return result

    def _enhance_eyes(
        self,
        depth: np.ndarray,
        masks,
        face_median: float,
        params: RefinementParams,
    ) -> np.ndarray:
        """
        Add controlled eye socket depth.

        Key insight: Eyes need to be SLIGHTLY recessed from the face surface
        to look natural, but NOT so deep they look hollow/dead.
        The sweet spot is a very subtle concavity.
        """
        result = depth.copy()

        eye_masks = [masks.left_eye, masks.right_eye]
        valid_eyes = []

        for eye_mask in eye_masks:
            if eye_mask.sum() < 10:
                valid_eyes.append(None)
                continue
            # Wide gaussian creates smooth natural socket shape
            center_mask = ndimage.gaussian_filter(eye_mask, sigma=8)
            center_mask = center_mask / max(center_mask.max(), 1e-6)
            valid_eyes.append(center_mask)

        # Balance left/right eye depth:
        # MiDaS often gives asymmetric depth due to head angle/lighting.
        # For bas-relief, we want symmetric eye treatment.
        # Average the current depth at both eyes and apply same recess to both.
        recess = 0.012 + 0.025 * params.eye_depth

        # Before applying recess, equalize the depth under both eyes
        eye_medians = []
        for i, eye_mask in enumerate(eye_masks):
            if valid_eyes[i] is not None and eye_mask.sum() > 10:
                eye_vals = result[eye_mask > 0.5]
                eye_medians.append(float(np.median(eye_vals)))
            else:
                eye_medians.append(None)

        # If both eyes detected, pull them toward the average depth
        if eye_medians[0] is not None and eye_medians[1] is not None:
            avg_eye = (eye_medians[0] + eye_medians[1]) / 2
            for i in range(2):
                if valid_eyes[i] is not None:
                    correction = avg_eye - eye_medians[i]
                    soft_mask = ndimage.gaussian_filter(eye_masks[i].astype(np.float64), sigma=10)
                    soft_mask = soft_mask / max(soft_mask.max(), 1e-6)
                    result = result + soft_mask * correction * 0.7  # 70% correction

        # Now apply uniform recess to both eyes
        for center_mask in valid_eyes:
            if center_mask is None:
                continue
            result = result - center_mask * recess

        # Add subtle brow ridge above each eye
        for eye_mask in eye_masks:
            if eye_mask.sum() < 10:
                continue
            brow_shift = max(3, int(eye_mask.shape[0] * 0.015))
            brow_mask = np.roll(eye_mask, -brow_shift, axis=0)
            brow_ridge = ndimage.gaussian_filter(brow_mask, sigma=8) * 0.005 * params.eye_depth
            result = result + brow_ridge

        return result

    def _enhance_mouth(
        self,
        depth: np.ndarray,
        masks,
        face_median: float,
        params: RefinementParams,
    ) -> np.ndarray:
        """Add subtle mouth/lip definition."""
        result = depth.copy()
        mouth_mask = masks.mouth

        if mouth_mask.sum() < 10:
            return result

        # Smooth mouth mask heavily to avoid polygon edge artifacts
        center_mask = ndimage.gaussian_filter(mouth_mask, sigma=6)
        center_mask = center_mask / max(center_mask.max(), 1e-6)

        # Very subtle recess for the lip area
        recess = 0.012 * params.mouth_depth
        result = result - center_mask * recess

        # Upper lip slightly more projected than lower lip
        upper_half = np.zeros_like(mouth_mask)
        mouth_rows = np.where(mouth_mask.sum(axis=1) > 0)[0]
        if len(mouth_rows) > 0:
            mid_row = mouth_rows[len(mouth_rows) // 2]
            upper_half[:mid_row, :] = mouth_mask[:mid_row, :]
            upper_boost = ndimage.gaussian_filter(upper_half, sigma=8) * 0.006 * params.mouth_depth
            result = result + upper_boost

        return result

    def _enhance_cheeks_jaw(
        self,
        depth: np.ndarray,
        masks,
        face_median: float,
        params: RefinementParams,
    ) -> np.ndarray:
        """Enhance cheek volume and jaw definition."""
        result = depth.copy()

        # Cheeks: add subtle convex volume
        for cheek_mask in [masks.left_cheek, masks.right_cheek]:
            if cheek_mask.sum() < 10:
                continue

            cheek_center = ndimage.gaussian_filter(cheek_mask, sigma=15)
            cheek_center = cheek_center / max(cheek_center.max(), 1e-6)

            # Subtle outward projection for cheek volume
            cheek_boost = cheek_center * 0.025 * params.cheek_volume
            result = result + cheek_boost

        # Jaw/chin: add definition by slightly recessing below jawline
        chin_mask = masks.chin
        if chin_mask.sum() > 10:
            chin_center = ndimage.gaussian_filter(chin_mask, sigma=10)
            chin_center = chin_center / max(chin_center.max(), 1e-6)
            # Slight chin projection
            chin_boost = chin_center * 0.018 * params.jaw_definition
            result = result + chin_boost

        return result

    def _enhance_forehead(
        self,
        depth: np.ndarray,
        masks,
        face_median: float,
        params: RefinementParams,
    ) -> np.ndarray:
        """Add natural forehead curvature."""
        result = depth.copy()
        forehead_mask = masks.forehead

        if forehead_mask.sum() < 10:
            return result

        # Forehead should have a gentle convex curve
        forehead_center = ndimage.gaussian_filter(forehead_mask, sigma=15)
        forehead_center = forehead_center / max(forehead_center.max(), 1e-6)

        # Gentle outward projection at center of forehead
        forehead_boost = forehead_center * 0.02 * params.forehead_roundness
        result = result + forehead_boost

        return result

    def _redistribute_face_depth(
        self,
        depth: np.ndarray,
        faces: list[FaceData],
        params: RefinementParams,
    ) -> np.ndarray:
        """
        Redistribute depth values so face features span a wider range.

        Problem: After MiDaS + enhancement, the face sits at 0.85-1.0 depth.
        The nose-to-eye range is only ~0.15. For a 6mm relief, that's only
        ~0.9mm of face variation — barely visible.

        Solution: Stretch the face region's depth to span 0.35-1.0, giving
        ~3.9mm of face feature depth. This makes the nose, eyes, and brow
        clearly visible in the carved relief.

        Uses a smooth blend so face doesn't have a hard boundary.
        """
        result = depth.copy()

        # Combine all face masks
        face_mask = np.zeros_like(depth)
        for face in faces:
            face_mask = np.maximum(face_mask, face.region_masks.face_outline)

        face_region = face_mask > 0.5
        if not face_region.any():
            return result

        face_values = depth[face_region]
        face_min = np.percentile(face_values, 2)
        face_max = np.percentile(face_values, 98)
        face_range = face_max - face_min

        if face_range < 0.01:
            return result

        # Target: stretch face features from their narrow band to span
        # target_min to target_max of the full depth range
        target_min = 0.35  # Eyes/lowest face features start here
        target_max = 1.0   # Nose tip at maximum

        # Create stretched version of the face
        # Map [face_min, face_max] -> [target_min, target_max]
        scale = (target_max - target_min) / face_range
        stretched = target_min + (depth - face_min) * scale

        # Clamp stretched values
        stretched = np.clip(stretched, 0, 1.0)

        # Background should stay at its original (low) values
        # Use a wide soft mask for smooth transition face → background
        transition_mask = ndimage.gaussian_filter(face_mask, sigma=15)
        transition_mask = np.clip(transition_mask, 0, 1)

        # Blend: face region uses stretched values, background keeps original
        result = depth * (1 - transition_mask) + stretched * transition_mask

        return result

    def _normalize(self, depth: np.ndarray) -> np.ndarray:
        """Normalize depth map to 0-1 range."""
        result = depth - depth.min()
        max_val = result.max()
        if max_val > 0:
            result = result / max_val
        return result.astype(np.float32)
