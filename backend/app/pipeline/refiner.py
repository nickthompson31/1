"""
Face-aware depth refinement — the "anti-creepy" engine, v3.

With Depth Anything V2, we get much better raw facial detail than MiDaS.
This refiner now focuses on:

1. GEOMETRIC SCULPTING: Use MediaPipe 478-point landmarks (which include Z depth)
   to construct a proper 3D face template, then blend it with the AI depth.
2. FEATURE ENHANCEMENT: Boost nose, define eye sockets, add brow ridges.
3. DEPTH REDISTRIBUTION: Stretch the face depth range for visible relief.
4. ANTI-CREEPY GUARDS: Prevent hollow eyes, spiky noses, grimace mouths.
"""

import numpy as np
from scipy import ndimage
from dataclasses import dataclass

from .face import FaceData


@dataclass
class RefinementParams:
    """User-adjustable parameters for depth refinement."""

    # Overall feature enhancement strength (0=subtle, 1=dramatic)
    feature_strength: float = 0.8

    # Eye socket depth (0=flush, 1=deep sockets)
    eye_depth: float = 0.5

    # Nose projection boost (0=natural, 1=strong projection)
    nose_projection: float = 0.85

    # Mouth/lip definition (0=flat, 1=defined)
    mouth_depth: float = 0.5

    # Forehead curvature emphasis
    forehead_roundness: float = 0.6

    # Cheek volume emphasis
    cheek_volume: float = 0.6

    # Jaw/chin definition
    jaw_definition: float = 0.6

    # Post-enhancement smoothing (0=crisp, 1=smooth)
    smoothing: float = 0.3

    # Detail preservation from original depth (0=sculpted only, 1=original detail)
    detail_level: float = 0.5

    # Background smoothing
    background_smoothing: float = 3.0


class DepthRefiner:
    """Sculpts and enhances facial depth for natural-looking bas-relief."""

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
            refined = ndimage.gaussian_filter(refined, sigma=1.0)
            return self._normalize(refined)

        # Build combined face mask
        face_mask = np.zeros_like(refined)
        for face in faces:
            face_mask = np.maximum(face_mask, face.region_masks.combined)

        # Step 1: Create geometric face sculpture from landmarks
        sculpture = self._build_face_sculpture(depth_map, faces, params)

        face_mask_soft = ndimage.gaussian_filter(face_mask, sigma=10)
        face_mask_soft = np.clip(face_mask_soft, 0, 1)

        # Step 2: Decompose AI depth into low-freq shape + high-freq detail
        # Low-freq = overall face dome shape (from AI)
        # High-freq = wrinkles, skin texture, hair strands, pores (from AI)
        ai_low = ndimage.gaussian_filter(depth_map, sigma=6.0)
        ai_detail = depth_map - ai_low  # This is the precious fine detail

        # Step 3: Decompose sculpture similarly
        sculpt_low = ndimage.gaussian_filter(sculpture, sigma=6.0)

        # Step 4: Combine: sculpture's low-freq structure + AI's fine detail
        # The sculpture provides nose ridge, eye sockets, brow ridges
        # The AI provides wrinkles, texture, hair, likeness
        blend = params.feature_strength * 0.7
        combined_low = ai_low * (1 - face_mask_soft * blend) + sculpt_low * face_mask_soft * blend

        # Add back ALL the AI fine detail — this is what gives us texture
        detail_strength = 0.5 + params.detail_level * 1.5  # 0.5-2.0x
        refined = combined_low + ai_detail * detail_strength

        # Step 5: Smooth background
        bg_smoothed = ndimage.gaussian_filter(refined, sigma=params.background_smoothing)
        refined = refined * face_mask_soft + bg_smoothed * (1 - face_mask_soft)

        # Step 6: Redistribute face depth for visible relief
        refined = self._redistribute_face_depth(refined, faces)

        # Step 7: Very light cleanup — just enough to blend sculpture edges
        if params.smoothing > 0:
            sigma = 0.3 + params.smoothing * 0.7  # Much lighter than before
            smoothed = ndimage.gaussian_filter(refined, sigma=sigma)
            refined = refined * (1 - face_mask_soft * 0.15) + smoothed * face_mask_soft * 0.15

        return self._normalize(refined)

    def _build_face_sculpture(
        self,
        depth_map: np.ndarray,
        faces: list[FaceData],
        params: RefinementParams,
    ) -> np.ndarray:
        """
        Build a geometric face sculpture using landmark positions.

        Instead of just enhancing the AI depth, we CONSTRUCT proper facial
        geometry from the 478 MediaPipe landmarks, creating:
        - A smooth dome base for the face
        - Nose ridge as the highest projection
        - Eye sockets as defined concavities
        - Brow ridges above the eyes
        - Cheek volume
        - Chin/jaw definition
        - Mouth area definition
        """
        sculpture = depth_map.copy()

        for face in faces:
            masks = face.region_masks
            lm = face.pixel_landmarks  # (478, 2) in pixel coords

            face_pixels = depth_map[masks.face_outline > 0.5]
            if len(face_pixels) == 0:
                continue

            face_median = np.median(face_pixels)
            face_max = np.percentile(face_pixels, 95)

            h, w = depth_map.shape

            # --- FACE DOME: Overall convex projection ---
            # This is what makes the face PROJECT OUT from the background
            # Without this, the face looks flat/sunken
            face_dome = ndimage.gaussian_filter(masks.face_outline, sigma=25)
            face_dome = face_dome / max(face_dome.max(), 1e-6)
            # Raise the dome shape — parabolic curve, highest at center
            face_dome = face_dome ** 0.7  # Broader dome shape
            dome_height = 0.15 * params.feature_strength
            sculpture = sculpture + face_dome * dome_height

            # --- NOSE: Strong central projection ---
            nose_boost = 0.30 * params.nose_projection
            # Use nose landmarks to find the ridge line
            nose_tip_idx = 1  # Nose tip in MediaPipe
            nose_bridge_idx = 6  # Bridge of nose
            if len(lm) > max(nose_tip_idx, nose_bridge_idx):
                tip = lm[nose_tip_idx]
                bridge = lm[nose_bridge_idx]

                # Create a gaussian mound centered on the nose
                yy, xx = np.ogrid[:h, :w]
                # Elliptical nose shape — taller than wide
                nose_cx = (tip[0] + bridge[0]) / 2
                nose_cy = (tip[1] + bridge[1]) / 2
                nose_length = abs(tip[1] - bridge[1])
                nose_width = nose_length * 0.5

                nose_dist = ((xx - nose_cx) / max(nose_width * 0.8, 1)) ** 2 + \
                            ((yy - nose_cy) / max(nose_length * 0.65, 1)) ** 2
                nose_mound = np.exp(-nose_dist * 2.0) * nose_boost

                # Strong tip peak
                tip_dist = ((xx - tip[0]) / max(nose_width * 0.5, 1)) ** 2 + \
                           ((yy - tip[1]) / max(nose_width * 0.4, 1)) ** 2
                tip_peak = np.exp(-tip_dist * 2.5) * nose_boost * 0.5

                sculpture = sculpture + nose_mound + tip_peak

            # --- EYES: Concave sockets with brow ridge ---
            eye_recess = 0.04 + 0.06 * params.eye_depth  # 0.04-0.10 (subtle!)
            for eye_indices_name in ["left_eye", "right_eye"]:
                eye_mask = getattr(masks, eye_indices_name)
                if eye_mask.sum() < 10:
                    continue

                # Find eye center from mask
                eye_rows = np.where(eye_mask.sum(axis=1) > 0)[0]
                eye_cols = np.where(eye_mask.sum(axis=0) > 0)[0]
                if len(eye_rows) == 0 or len(eye_cols) == 0:
                    continue

                eye_cy = (eye_rows[0] + eye_rows[-1]) / 2
                eye_cx = (eye_cols[0] + eye_cols[-1]) / 2
                eye_ry = (eye_rows[-1] - eye_rows[0]) / 2
                eye_rx = (eye_cols[-1] - eye_cols[0]) / 2

                # Elliptical socket depression
                yy, xx = np.ogrid[:h, :w]
                eye_dist = ((xx - eye_cx) / max(eye_rx * 1.0, 1)) ** 2 + \
                           ((yy - eye_cy) / max(eye_ry * 1.0, 1)) ** 2
                socket = np.exp(-eye_dist * 2.2) * eye_recess
                sculpture = sculpture - socket

                # Brow ridge: project upward just above the eye
                brow_cy = eye_cy - eye_ry * 1.3
                brow_dist = ((xx - eye_cx) / max(eye_rx * 1.5, 1)) ** 2 + \
                            ((yy - brow_cy) / max(eye_ry * 0.6, 1)) ** 2
                brow = np.exp(-brow_dist * 2.0) * eye_recess * 0.8
                sculpture = sculpture + brow

            # --- MOUTH: Subtle lip definition ---
            mouth_mask = masks.mouth
            if mouth_mask.sum() > 10:
                mouth_rows = np.where(mouth_mask.sum(axis=1) > 0)[0]
                mouth_cols = np.where(mouth_mask.sum(axis=0) > 0)[0]
                if len(mouth_rows) > 0 and len(mouth_cols) > 0:
                    mouth_cy = (mouth_rows[0] + mouth_rows[-1]) / 2
                    mouth_cx = (mouth_cols[0] + mouth_cols[-1]) / 2
                    mouth_ry = (mouth_rows[-1] - mouth_rows[0]) / 2
                    mouth_rx = (mouth_cols[-1] - mouth_cols[0]) / 2

                    yy, xx = np.ogrid[:h, :w]
                    # Lip crease — narrow horizontal recess
                    lip_dist = ((xx - mouth_cx) / max(mouth_rx * 1.2, 1)) ** 2 + \
                               ((yy - mouth_cy) / max(mouth_ry * 0.5, 1)) ** 2
                    lip_crease = np.exp(-lip_dist * 2.5) * 0.04 * params.mouth_depth
                    sculpture = sculpture - lip_crease

                    # Upper lip projection
                    upper_cy = mouth_cy - mouth_ry * 0.5
                    upper_dist = ((xx - mouth_cx) / max(mouth_rx * 0.8, 1)) ** 2 + \
                                 ((yy - upper_cy) / max(mouth_ry * 0.4, 1)) ** 2
                    upper_lip = np.exp(-upper_dist * 3.0) * 0.025 * params.mouth_depth
                    sculpture = sculpture + upper_lip

            # --- CHEEKS: Subtle volume ---
            for cheek_mask in [masks.left_cheek, masks.right_cheek]:
                if cheek_mask.sum() < 10:
                    continue
                cheek_smooth = ndimage.gaussian_filter(cheek_mask, sigma=20)
                cheek_smooth = cheek_smooth / max(cheek_smooth.max(), 1e-6)
                sculpture = sculpture + cheek_smooth * 0.05 * params.cheek_volume

            # --- FOREHEAD: Dome curvature ---
            if masks.forehead.sum() > 10:
                forehead_smooth = ndimage.gaussian_filter(masks.forehead, sigma=20)
                forehead_smooth = forehead_smooth / max(forehead_smooth.max(), 1e-6)
                sculpture = sculpture + forehead_smooth * 0.04 * params.forehead_roundness

            # --- CHIN: Definition ---
            if masks.chin.sum() > 10:
                chin_smooth = ndimage.gaussian_filter(masks.chin, sigma=12)
                chin_smooth = chin_smooth / max(chin_smooth.max(), 1e-6)
                sculpture = sculpture + chin_smooth * 0.035 * params.jaw_definition

        return sculpture

    def _redistribute_face_depth(
        self,
        depth: np.ndarray,
        faces: list[FaceData],
    ) -> np.ndarray:
        """Stretch face depth range for visible relief."""
        result = depth.copy()

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

        # Target: face features span 0.45-1.0 of depth range
        # Higher min = face projects MORE above background
        target_min = 0.45
        target_max = 1.0
        scale = (target_max - target_min) / face_range
        stretched = target_min + (depth - face_min) * scale
        stretched = np.clip(stretched, 0, 1.0)

        # Smooth transition mask
        transition_mask = ndimage.gaussian_filter(face_mask, sigma=15)
        transition_mask = np.clip(transition_mask, 0, 1)

        result = depth * (1 - transition_mask) + stretched * transition_mask
        return result

    def _normalize(self, depth: np.ndarray) -> np.ndarray:
        result = depth - depth.min()
        max_val = result.max()
        if max_val > 0:
            result = result / max_val
        return result.astype(np.float32)
