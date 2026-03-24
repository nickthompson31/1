"""
Pipeline orchestrator — ties all processing stages together.

Provides a simple, high-level API:
    pipeline = PortraitPipeline()
    result = pipeline.process(image, quality="preview")
    result.save_stl("output.stl")
"""

import time
import numpy as np
from PIL import Image
from dataclasses import dataclass, field
from stl import mesh as stl_mesh

from .preprocessor import Preprocessor
from .depth import DepthEstimator
from .face import FaceDetector, FaceData
from .refiner import DepthRefiner, RefinementParams
from .relief import ReliefMapper, ReliefParams, ReliefStyle
from .mesh import MeshGenerator


@dataclass
class PipelineResult:
    """Complete results from the portrait processing pipeline."""

    # The generated 3D mesh
    mesh: stl_mesh.Mesh

    # Intermediate results (useful for debugging and UI preview)
    preprocessed_image: Image.Image
    raw_depth_map: np.ndarray
    refined_depth_map: np.ndarray
    relief_map: np.ndarray
    faces_detected: list[FaceData]
    background_mask: np.ndarray | None

    # Timing info
    timings: dict = field(default_factory=dict)

    def save_stl(self, filepath: str, binary: bool = True):
        """Save the mesh to an STL file."""
        generator = MeshGenerator()
        generator.save_stl(self.mesh, filepath, binary)

    def stl_bytes(self) -> bytes:
        """Get STL as bytes (for API responses)."""
        generator = MeshGenerator()
        return generator.to_bytes(self.mesh)

    def depth_preview(self) -> Image.Image:
        """Get the refined depth map as a grayscale image (for UI preview)."""
        depth_uint8 = (self.refined_depth_map * 255).astype(np.uint8)
        return Image.fromarray(depth_uint8, mode="L")

    def relief_preview(self) -> Image.Image:
        """Get the relief map as a grayscale image."""
        normalized = self.relief_map - self.relief_map.min()
        max_val = normalized.max()
        if max_val > 0:
            normalized = normalized / max_val
        relief_uint8 = (normalized * 255).astype(np.uint8)
        return Image.fromarray(relief_uint8, mode="L")


class PortraitPipeline:
    """
    Main pipeline for converting portrait photos to carve-ready STL files.

    Usage:
        pipeline = PortraitPipeline()

        # Quick preview
        result = pipeline.process(image, quality="preview")

        # High quality final
        result = pipeline.process(image, quality="final")

        # Save output
        result.save_stl("portrait.stl")
    """

    def __init__(self):
        self.preprocessor = Preprocessor()
        self.depth_estimator = DepthEstimator()
        self.face_detector = FaceDetector()
        self.refiner = DepthRefiner()
        self.relief_mapper = ReliefMapper()
        self.mesh_generator = MeshGenerator()

    def process(
        self,
        image: Image.Image,
        quality: str = "preview",
        refinement_params: RefinementParams | None = None,
        relief_params: ReliefParams | None = None,
        remove_background: bool = True,
        mesh_resolution: int | None = None,
    ) -> PipelineResult:
        """
        Process a portrait image through the full pipeline.

        Args:
            image: Input portrait photo (PIL Image)
            quality: "preview" (fast) or "final" (high quality)
            refinement_params: Face-aware depth refinement settings
            relief_params: Relief curve and output settings
            remove_background: Whether to remove background
            mesh_resolution: Max mesh grid dimension. None = auto based on quality.

        Returns:
            PipelineResult with mesh and all intermediate data.
        """
        if refinement_params is None:
            refinement_params = RefinementParams()
        if relief_params is None:
            relief_params = ReliefParams()

        # Auto mesh resolution based on quality
        if mesh_resolution is None:
            mesh_resolution = 256 if quality == "preview" else 512

        timings = {}

        # --- Stage 1: Preprocessing ---
        t0 = time.time()
        processed_image, bg_mask = self.preprocessor.process(
            image,
            remove_background=remove_background,
            max_dimension=768 if quality == "preview" else 1024,
        )
        timings["preprocessing"] = time.time() - t0

        # --- Stage 2: Face Detection ---
        t0 = time.time()
        faces = self.face_detector.detect(processed_image)
        timings["face_detection"] = time.time() - t0

        # --- Stage 3: Depth Estimation ---
        t0 = time.time()
        target_size = 384 if quality == "preview" else None
        raw_depth = self.depth_estimator.estimate(
            processed_image,
            quality=quality,
            target_size=target_size,
        )
        timings["depth_estimation"] = time.time() - t0

        # Resize depth map to match processed image if needed
        img_h, img_w = np.array(processed_image).shape[:2]
        if raw_depth.shape != (img_h, img_w):
            from scipy import ndimage as ndi
            raw_depth = ndi.zoom(
                raw_depth,
                (img_h / raw_depth.shape[0], img_w / raw_depth.shape[1]),
                order=3,
            )

        # Apply background mask to depth (set background to 0)
        if bg_mask is not None:
            # Resize mask if needed
            if bg_mask.shape != raw_depth.shape:
                bg_mask = ndi.zoom(
                    bg_mask,
                    (raw_depth.shape[0] / bg_mask.shape[0],
                     raw_depth.shape[1] / bg_mask.shape[1]),
                    order=1,
                )
            raw_depth = raw_depth * bg_mask

        # --- Stage 4: Face-Aware Refinement ---
        t0 = time.time()
        refined_depth = self.refiner.refine(raw_depth, faces, refinement_params)
        timings["refinement"] = time.time() - t0

        # Re-apply background mask after refinement
        if bg_mask is not None:
            refined_depth = refined_depth * bg_mask

        # --- Stage 5: Artistic Relief Curve ---
        t0 = time.time()
        relief_map = self.relief_mapper.apply(refined_depth, relief_params)
        timings["relief_mapping"] = time.time() - t0

        # --- Stage 6: Mesh Generation ---
        t0 = time.time()
        mesh = self.mesh_generator.generate(
            relief_map,
            width_mm=relief_params.output_width_mm or 150.0,
            height_mm=relief_params.output_height_mm,
            base_thickness_mm=relief_params.base_thickness_mm,
            mesh_resolution=mesh_resolution,
        )
        timings["mesh_generation"] = time.time() - t0

        timings["total"] = sum(timings.values())

        return PipelineResult(
            mesh=mesh,
            preprocessed_image=processed_image,
            raw_depth_map=raw_depth,
            refined_depth_map=refined_depth,
            relief_map=relief_map,
            faces_detected=faces,
            background_mask=bg_mask,
            timings=timings,
        )
