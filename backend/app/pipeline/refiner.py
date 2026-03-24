"""
Face-aware depth refinement — the "classical sculpture" engine, v4.

Designed to produce output that looks like beautiful Roman bas-relief sculpture.
Classical portraits in marble/stone have:

1. IDEALIZED FORMS: Smooth surfaces, no skin texture or pore noise.
2. STRONG BONE STRUCTURE: Prominent brow ridge, nose bridge, cheekbones.
3. GRACEFUL TRANSITIONS: No harsh edges between features.
4. DRAMATIC PROJECTION: Face projects boldly from the background plane.

Pipeline:
1. GEOMETRIC SCULPTING: Construct idealized facial geometry from landmarks.
2. FEATURE ENHANCEMENT: Bold nose bridge, orbital rims, brow ridges, cheekbones.
3. PROGRESSIVE SMOOTHING: Multi-pass smoothing for marble-like surfaces.
4. DEPTH REDISTRIBUTION: Stretch face depth range for visible, dramatic relief.
"""

import numpy as np
from scipy import ndimage
from dataclasses import dataclass

from .face import FaceData


@dataclass
class RefinementParams:
    """User-adjustable parameters for depth refinement."""

    # Overall feature enhancement strength (0=subtle, 1=dramatic)
    feature_strength: float = 0.9

    # Eye socket depth (0=flush, 1=deep sockets)
    eye_depth: float = 0.6

    # Nose projection boost (0=natural, 1=strong projection)
    nose_projection: float = 0.95

    # Mouth/lip definition (0=flat, 1=defined)
    mouth_depth: float = 0.6

    # Forehead curvature emphasis
    forehead_roundness: float = 0.75

    # Cheek volume emphasis
    cheek_volume: float = 0.75

    # Jaw/chin definition
    jaw_definition: float = 0.7

    # Post-enhancement smoothing (0=crisp, 1=smooth like marble)
    smoothing: float = 0.7

    # Detail preservation from original depth (0=sculpted only, 1=original detail)
    # Lower = smoother, more idealized (like classical sculpture)
    detail_level: float = 0.25

    # Background smoothing
    background_smoothing: float = 5.0


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
            refined = ndimage.gaussian_filter(refined, sigma=2.0)
            return self._normalize(refined)

        # Build combined face mask
        face_mask = np.zeros_like(refined)
        for face in faces:
            face_mask = np.maximum(face_mask, face.region_masks.combined)

        # Wider soft mask for smoother transitions (classical sculpture has no hard edges)
        face_mask_soft = ndimage.gaussian_filter(face_mask, sigma=15)
        face_mask_soft = np.clip(face_mask_soft, 0, 1)

        # === APPROACH: Smooth stone base + crisp sculpted features ===
        # Like a real sculptor: start with a smooth block, then carve features.

        # Step 1: Create a smooth AI base (remove texture, keep structure)
        ai_low = ndimage.gaussian_filter(depth_map, sigma=8.0)
        ai_detail = depth_map - ai_low
        # Marble-smooth the detail layer to kill skin texture
        smoothed_detail = self._marble_smooth(ai_detail, face_mask_soft, params.smoothing)
        detail_strength = 0.15 + params.detail_level * 0.7
        smooth_base = ai_low + smoothed_detail * detail_strength

        # Step 2: Build geometric face sculpture
        # _build_face_sculpture returns depth_map + sculpted features
        sculpture = self._build_face_sculpture(depth_map, faces, params)
        # Extract JUST the sculpted additions (nose, brow, sockets, etc.)
        sculpt_additions = sculpture - depth_map
        # Light smoothing for graceful transitions, but keep features CRISP
        sculpt_additions = ndimage.gaussian_filter(sculpt_additions, sigma=2.0)

        # Step 3: Apply sculpted features directly onto the smooth base
        # This is the key — features go on AFTER smoothing, so they survive
        blend = params.feature_strength * 0.9
        refined = smooth_base + sculpt_additions * face_mask_soft * blend

        # Step 4: Smooth background aggressively
        bg_smoothed = ndimage.gaussian_filter(refined, sigma=params.background_smoothing)
        refined = refined * face_mask_soft + bg_smoothed * (1 - face_mask_soft)

        # Step 5: Redistribute face depth for dramatic projection
        refined = self._redistribute_face_depth(refined, faces)

        return self._normalize(refined)

    def _marble_smooth(
        self,
        depth: np.ndarray,
        face_mask: np.ndarray,
        smoothing: float,
    ) -> np.ndarray:
        """
        Multi-pass progressive smoothing for marble/stone-carved appearance.

        Classical sculpture has smooth, flowing surfaces with gentle gradients
        between features. We apply smoothing at multiple scales:
        - Fine pass: removes skin texture and noise
        - Medium pass: softens transitions between features
        - Broad pass: very gentle, ensures overall surface flow
        """
        result = depth.copy()

        # Fine smoothing — removes texture noise (pores, wrinkles)
        fine_sigma = 1.0 + smoothing * 1.5  # 1.0 - 2.5
        fine_smoothed = ndimage.gaussian_filter(result, sigma=fine_sigma)
        # Apply strongly within face region
        fine_blend = 0.5 + smoothing * 0.4  # 0.5 - 0.9
        result = result * (1 - face_mask * fine_blend) + fine_smoothed * face_mask * fine_blend

        # Medium smoothing — softens feature transitions
        med_sigma = 2.5 + smoothing * 2.0  # 2.5 - 4.5
        med_smoothed = ndimage.gaussian_filter(result, sigma=med_sigma)
        med_blend = 0.15 + smoothing * 0.2  # 0.15 - 0.35
        result = result * (1 - face_mask * med_blend) + med_smoothed * face_mask * med_blend

        # Broad smoothing — ensures overall surface flow
        broad_sigma = 5.0 + smoothing * 3.0  # 5.0 - 8.0
        broad_smoothed = ndimage.gaussian_filter(result, sigma=broad_sigma)
        broad_blend = 0.05 + smoothing * 0.1  # 0.05 - 0.15
        result = result * (1 - face_mask * broad_blend) + broad_smoothed * face_mask * broad_blend

        return result

    def _build_face_sculpture(
        self,
        depth_map: np.ndarray,
        faces: list[FaceData],
        params: RefinementParams,
    ) -> np.ndarray:
        """
        Build an idealized face sculpture inspired by classical Roman portraiture.

        Roman bas-reliefs feature:
        - Bold, smooth face dome projecting from the background
        - Strong continuous nose bridge (from brow to tip)
        - Defined orbital rims and prominent brow ridge
        - High, structured cheekbones
        - Idealized forehead curvature
        - Clean jawline and chin
        """
        sculpture = depth_map.copy()

        for face in faces:
            masks = face.region_masks
            lm = face.pixel_landmarks  # (478, 2) in pixel coords

            face_pixels = depth_map[masks.face_outline > 0.5]
            if len(face_pixels) == 0:
                continue

            h, w = depth_map.shape

            # --- FACE DOME: Bold convex projection ---
            face_dome = ndimage.gaussian_filter(masks.face_outline, sigma=30)
            face_dome = face_dome / max(face_dome.max(), 1e-6)
            face_dome = face_dome ** 0.6
            dome_height = 0.28 * params.feature_strength
            sculpture = sculpture + face_dome * dome_height

            # --- NOSE: Strong continuous bridge (the Roman nose) ---
            # Classical Roman portraits have a strong, straight nose bridge
            # running continuously from the brow line to the tip
            nose_boost = 0.38 * params.nose_projection
            nose_tip_idx = 1
            nose_bridge_idx = 6
            if len(lm) > max(nose_tip_idx, nose_bridge_idx):
                tip = lm[nose_tip_idx]
                bridge = lm[nose_bridge_idx]

                yy, xx = np.ogrid[:h, :w]

                # Nose center and dimensions
                nose_cx = (tip[0] + bridge[0]) / 2
                nose_cy = (tip[1] + bridge[1]) / 2
                nose_length = abs(tip[1] - bridge[1])
                nose_width = nose_length * 0.45

                # Main nose mound — smooth elliptical
                nose_dist = ((xx - nose_cx) / max(nose_width * 0.7, 1)) ** 2 + \
                            ((yy - nose_cy) / max(nose_length * 0.7, 1)) ** 2
                nose_mound = np.exp(-nose_dist * 1.8) * nose_boost

                # Continuous bridge line from brow to tip
                # This is THE defining feature of a Roman profile
                bridge_top_y = bridge[1] - nose_length * 0.3  # Extend into brow
                bridge_cy = (bridge_top_y + tip[1]) / 2
                bridge_len = abs(tip[1] - bridge_top_y)
                bridge_dist = ((xx - nose_cx) / max(nose_width * 0.35, 1)) ** 2 + \
                              ((yy - bridge_cy) / max(bridge_len * 0.55, 1)) ** 2
                bridge_line = np.exp(-bridge_dist * 2.0) * nose_boost * 0.6

                # Rounded tip
                tip_dist = ((xx - tip[0]) / max(nose_width * 0.55, 1)) ** 2 + \
                           ((yy - tip[1]) / max(nose_width * 0.45, 1)) ** 2
                tip_peak = np.exp(-tip_dist * 2.0) * nose_boost * 0.45

                sculpture = sculpture + nose_mound + bridge_line + tip_peak

            # --- EYES: Defined orbital sockets with strong brow ridge ---
            # Classical sculpture has clearly defined orbital rims, not just
            # holes — the eye sits in a smooth, bowl-shaped depression with
            # a pronounced ridge above
            eye_recess = 0.08 + 0.14 * params.eye_depth  # 0.08-0.22
            for eye_indices_name in ["left_eye", "right_eye"]:
                eye_mask = getattr(masks, eye_indices_name)
                if eye_mask.sum() < 10:
                    continue

                eye_rows = np.where(eye_mask.sum(axis=1) > 0)[0]
                eye_cols = np.where(eye_mask.sum(axis=0) > 0)[0]
                if len(eye_rows) == 0 or len(eye_cols) == 0:
                    continue

                eye_cy = (eye_rows[0] + eye_rows[-1]) / 2
                eye_cx = (eye_cols[0] + eye_cols[-1]) / 2
                eye_ry = (eye_rows[-1] - eye_rows[0]) / 2
                eye_rx = (eye_cols[-1] - eye_cols[0]) / 2

                yy, xx = np.ogrid[:h, :w]

                # Orbital socket — smooth bowl shape (wider than the eye itself)
                socket_dist = ((xx - eye_cx) / max(eye_rx * 1.3, 1)) ** 2 + \
                              ((yy - eye_cy) / max(eye_ry * 1.2, 1)) ** 2
                socket = np.exp(-socket_dist * 1.8) * eye_recess
                sculpture = sculpture - socket

                # Strong brow ridge — the hallmark of classical sculpture
                brow_cy = eye_cy - eye_ry * 1.5
                brow_dist = ((xx - eye_cx) / max(eye_rx * 1.8, 1)) ** 2 + \
                            ((yy - brow_cy) / max(eye_ry * 0.7, 1)) ** 2
                brow = np.exp(-brow_dist * 1.6) * eye_recess * 1.4
                sculpture = sculpture + brow

                # Orbital rim — ridge around the socket
                rim_dist = ((xx - eye_cx) / max(eye_rx * 1.5, 1)) ** 2 + \
                           ((yy - eye_cy) / max(eye_ry * 1.4, 1)) ** 2
                rim_ring = np.exp(-((rim_dist - 1.0) ** 2) * 3.0) * eye_recess * 0.4
                sculpture = sculpture + rim_ring

            # --- MOUTH: Classical lip definition ---
            # Roman portraits have clearly defined but smooth lips with
            # a philtrum (the groove above the upper lip)
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

                    # Lip separation crease
                    lip_dist = ((xx - mouth_cx) / max(mouth_rx * 1.2, 1)) ** 2 + \
                               ((yy - mouth_cy) / max(mouth_ry * 0.4, 1)) ** 2
                    lip_crease = np.exp(-lip_dist * 2.5) * 0.07 * params.mouth_depth
                    sculpture = sculpture - lip_crease

                    # Upper lip — projects forward (classical Cupid's bow)
                    upper_cy = mouth_cy - mouth_ry * 0.6
                    upper_dist = ((xx - mouth_cx) / max(mouth_rx * 0.9, 1)) ** 2 + \
                                 ((yy - upper_cy) / max(mouth_ry * 0.5, 1)) ** 2
                    upper_lip = np.exp(-upper_dist * 2.5) * 0.06 * params.mouth_depth
                    sculpture = sculpture + upper_lip

                    # Lower lip — subtle forward projection
                    lower_cy = mouth_cy + mouth_ry * 0.4
                    lower_dist = ((xx - mouth_cx) / max(mouth_rx * 0.85, 1)) ** 2 + \
                                 ((yy - lower_cy) / max(mouth_ry * 0.45, 1)) ** 2
                    lower_lip = np.exp(-lower_dist * 2.5) * 0.04 * params.mouth_depth
                    sculpture = sculpture + lower_lip

                    # Philtrum — subtle vertical groove above upper lip
                    philtrum_cy = mouth_cy - mouth_ry * 1.5
                    philtrum_dist = ((xx - mouth_cx) / max(mouth_rx * 0.15, 1)) ** 2 + \
                                   ((yy - philtrum_cy) / max(mouth_ry * 0.8, 1)) ** 2
                    philtrum = np.exp(-philtrum_dist * 2.0) * 0.015 * params.mouth_depth
                    sculpture = sculpture - philtrum

            # --- CHEEKBONES: Prominent bone structure ---
            # Classical sculptures emphasize the zygomatic arch (cheekbone)
            for cheek_mask in [masks.left_cheek, masks.right_cheek]:
                if cheek_mask.sum() < 10:
                    continue
                cheek_smooth = ndimage.gaussian_filter(cheek_mask, sigma=22)
                cheek_smooth = cheek_smooth / max(cheek_smooth.max(), 1e-6)
                # More pronounced cheekbone projection
                sculpture = sculpture + cheek_smooth * 0.12 * params.cheek_volume

                # Hollow below the cheekbone for strong bone structure definition
                cheek_rows = np.where(cheek_mask.sum(axis=1) > 0)[0]
                if len(cheek_rows) > 5:
                    lower_cheek = cheek_mask.copy()
                    mid_row = (cheek_rows[0] + cheek_rows[-1]) // 2
                    lower_cheek[:mid_row, :] = 0
                    lower_smooth = ndimage.gaussian_filter(lower_cheek.astype(np.float64), sigma=15)
                    lower_smooth = lower_smooth / max(lower_smooth.max(), 1e-6)
                    sculpture = sculpture - lower_smooth * 0.05 * params.cheek_volume

            # --- FOREHEAD: Smooth dome curvature ---
            if masks.forehead.sum() > 10:
                forehead_smooth = ndimage.gaussian_filter(masks.forehead, sigma=25)
                forehead_smooth = forehead_smooth / max(forehead_smooth.max(), 1e-6)
                # Broader, smoother dome
                forehead_smooth = forehead_smooth ** 0.8
                sculpture = sculpture + forehead_smooth * 0.09 * params.forehead_roundness

            # --- CHIN/JAW: Clean definition ---
            if masks.chin.sum() > 10:
                chin_smooth = ndimage.gaussian_filter(masks.chin, sigma=15)
                chin_smooth = chin_smooth / max(chin_smooth.max(), 1e-6)
                sculpture = sculpture + chin_smooth * 0.07 * params.jaw_definition

        return sculpture

    def _redistribute_face_depth(
        self,
        depth: np.ndarray,
        faces: list[FaceData],
    ) -> np.ndarray:
        """Stretch face depth range for dramatic, bold relief projection."""
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

        # Target: face projects boldly — spanning 0.35-1.0 of depth range
        # Wider range = more dramatic relief with better shadow play
        target_min = 0.35
        target_max = 1.0
        scale = (target_max - target_min) / face_range
        stretched = target_min + (depth - face_min) * scale
        stretched = np.clip(stretched, 0, 1.0)

        # Wide, smooth transition for graceful falloff
        transition_mask = ndimage.gaussian_filter(face_mask, sigma=20)
        transition_mask = np.clip(transition_mask, 0, 1)

        result = depth * (1 - transition_mask) + stretched * transition_mask
        return result

    def _normalize(self, depth: np.ndarray) -> np.ndarray:
        result = depth - depth.min()
        max_val = result.max()
        if max_val > 0:
            result = result / max_val
        return result.astype(np.float32)
