"""
Artistic bas-relief depth curve mapping.

Real face depth spans ~6 inches. In bas-relief, this compresses to 0.25-0.5".
A LINEAR compression looks flat and lifeless.

Instead, we use carefully designed transfer curves that:
1. Compress background depths (they don't matter much)
2. EXPAND mid-range depths (where facial features live)
3. Cap foreground projection (nose tip shouldn't spike)

This creates the "hand-carved" aesthetic that makes relief portraits look natural.
"""

import numpy as np
from scipy import ndimage
from dataclasses import dataclass
from enum import Enum


class ReliefStyle(str, Enum):
    """Pre-defined artistic relief styles."""

    CLASSICAL = "classical"  # Traditional portrait relief, balanced depth
    DRAMATIC = "dramatic"  # High contrast, deep shadows
    SUBTLE = "subtle"  # Gentle, low relief — good for large pieces
    COIN = "coin"  # Very shallow, coin/medallion style


@dataclass
class ReliefParams:
    """Parameters controlling the relief curve and output."""

    # Maximum relief depth in mm (physical carving depth)
    max_depth_mm: float = 6.0

    # Relief style preset
    style: ReliefStyle = ReliefStyle.CLASSICAL

    # Curve gamma: <1.0 = expand highlights (more face detail),
    # >1.0 = expand shadows (more background detail)
    gamma: float = 0.6

    # Mid-tone expansion: how much to stretch the middle depth range
    midtone_boost: float = 1.5

    # Background compression: how aggressively to flatten the background
    background_compression: float = 0.8

    # Border/frame options
    add_border: bool = False
    border_width_mm: float = 3.0
    border_depth_mm: float = 1.0

    # Base plate thickness in mm
    base_thickness_mm: float = 2.0

    # Output dimensions (mm). None = auto from image aspect ratio
    output_width_mm: float | None = 150.0  # ~6 inches
    output_height_mm: float | None = None  # auto from aspect ratio

    # Vignette: fade depth to zero at edges
    vignette: bool = True
    vignette_strength: float = 0.8

    # Invert relief (cut into surface vs. raised from surface)
    invert: bool = False


# Style presets
STYLE_PRESETS = {
    ReliefStyle.CLASSICAL: {
        "gamma": 0.55,
        "midtone_boost": 1.4,
        "background_compression": 0.7,
        "max_depth_mm": 6.0,
    },
    ReliefStyle.DRAMATIC: {
        "gamma": 0.4,
        "midtone_boost": 1.8,
        "background_compression": 0.9,
        "max_depth_mm": 8.0,
    },
    ReliefStyle.SUBTLE: {
        "gamma": 0.7,
        "midtone_boost": 1.2,
        "background_compression": 0.5,
        "max_depth_mm": 3.0,
    },
    ReliefStyle.COIN: {
        "gamma": 0.75,
        "midtone_boost": 1.1,
        "background_compression": 0.9,
        "max_depth_mm": 1.5,
    },
}


class ReliefMapper:
    """Maps refined depth to artistic bas-relief depth profile."""

    def apply(
        self,
        depth_map: np.ndarray,
        params: ReliefParams | None = None,
    ) -> np.ndarray:
        """
        Apply artistic relief curve to a refined depth map.

        Args:
            depth_map: Refined depth map (0-1, float32)
            params: Relief parameters

        Returns:
            Relief depth map in mm (float32). Values represent physical
            carving depth above the base plane.
        """
        if params is None:
            params = ReliefParams()

        # Apply style preset defaults (user params override)
        preset = STYLE_PRESETS.get(params.style, {})

        gamma = params.gamma
        midtone_boost = params.midtone_boost
        bg_compression = params.background_compression
        max_depth = params.max_depth_mm

        # Start with normalized depth (0-1)
        relief = depth_map.copy().astype(np.float64)

        # Step 1: Apply the artistic transfer curve
        relief = self._apply_transfer_curve(relief, gamma, midtone_boost, bg_compression)

        # Step 2: Apply vignette if requested
        if params.vignette:
            relief = self._apply_vignette(relief, params.vignette_strength)

        # Step 3: Invert if requested (for intaglio/cut-in style)
        if params.invert:
            relief = 1.0 - relief

        # Step 4: Scale to physical depth (mm)
        relief = relief * max_depth

        return relief.astype(np.float32)

    def _apply_transfer_curve(
        self,
        depth: np.ndarray,
        gamma: float,
        midtone_boost: float,
        bg_compression: float,
    ) -> np.ndarray:
        """
        Apply a multi-stage transfer curve that mimics hand-carved relief.

        The curve has three zones:
        1. Shadows (background): compressed — background doesn't need detail
        2. Midtones (face features): expanded — this is where the portrait lives
        3. Highlights (nose tip, etc.): gently compressed — prevent spiky peaks
        """
        result = depth.copy()

        # Stage 1: Gamma curve — overall tonal redistribution
        # gamma < 1.0 expands highlights (face features)
        result = np.power(np.clip(result, 0, 1), gamma)

        # Stage 2: S-curve for midtone expansion
        # This is like Photoshop's "curves" — stretch the middle, compress extremes
        if midtone_boost != 1.0:
            result = self._sigmoid_midtone_boost(result, midtone_boost)

        # Stage 3: Background compression
        # Push low values (background) closer to zero
        if bg_compression > 0:
            threshold = 0.3  # Below this = "background"
            bg_mask = np.clip(1.0 - result / threshold, 0, 1)
            compression_factor = 1.0 - bg_compression * 0.5
            result = result * (1.0 - bg_mask) + result * compression_factor * bg_mask

        # Re-normalize
        result = result - result.min()
        max_val = result.max()
        if max_val > 0:
            result = result / max_val

        return result

    def _sigmoid_midtone_boost(
        self, depth: np.ndarray, strength: float
    ) -> np.ndarray:
        """
        Apply an S-shaped curve that expands midtones.

        Uses a modified sigmoid centered at 0.5 depth.
        Strength controls how aggressive the midtone expansion is.
        """
        # Steepness of the S-curve
        k = 4.0 + (strength - 1.0) * 8.0  # strength 1.0 → k=4, 2.0 → k=12

        # Centered sigmoid
        result = 1.0 / (1.0 + np.exp(-k * (depth - 0.5)))

        # Re-normalize to 0-1
        result = result - result.min()
        max_val = result.max()
        if max_val > 0:
            result = result / max_val

        return result

    def _apply_vignette(
        self, depth: np.ndarray, strength: float
    ) -> np.ndarray:
        """
        Apply oval vignette that fades depth to zero at edges.

        This creates a natural "fading" border rather than a hard edge,
        which looks much better when carved and avoids sharp depth cliffs.
        """
        h, w = depth.shape

        # Create coordinate grids
        y = np.linspace(-1, 1, h)
        x = np.linspace(-1, 1, w)
        xx, yy = np.meshgrid(x, y)

        # Oval distance from center
        dist = np.sqrt(xx ** 2 + yy ** 2)

        # Smooth falloff using a cosine curve
        inner_radius = 0.6  # Start fading here
        outer_radius = 1.0  # Fully faded here

        vignette = np.ones_like(dist)
        fade_zone = (dist > inner_radius) & (dist <= outer_radius)
        t = (dist[fade_zone] - inner_radius) / (outer_radius - inner_radius)
        vignette[fade_zone] = 0.5 * (1 + np.cos(np.pi * t))  # Smooth cosine fade
        vignette[dist > outer_radius] = 0

        # Apply strength
        vignette = 1.0 - strength * (1.0 - vignette)

        return depth * vignette
