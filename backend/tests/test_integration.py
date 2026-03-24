#!/usr/bin/env python3
"""
Integration tests for the portrait STL pipeline.

Tests each stage individually, then the full pipeline end-to-end.
Uses a synthetic gradient image (no real photo needed).
"""

import numpy as np
from PIL import Image, ImageDraw
import tempfile
import os

from app.pipeline.preprocessor import Preprocessor
from app.pipeline.depth import DepthEstimator
from app.pipeline.face import FaceDetector
from app.pipeline.refiner import DepthRefiner, RefinementParams
from app.pipeline.relief import ReliefMapper, ReliefParams, ReliefStyle
from app.pipeline.mesh import MeshGenerator
from app.pipeline.orchestrator import PortraitPipeline


def create_test_image(width=400, height=500) -> Image.Image:
    """Create a simple synthetic image for testing.
    Not a real face, but exercises the pipeline code paths."""
    img = Image.new("RGB", (width, height), (200, 180, 160))
    draw = ImageDraw.Draw(img)

    # Oval "face" shape
    cx, cy = width // 2, height // 2
    face_rx, face_ry = width // 3, height // 3
    draw.ellipse(
        [cx - face_rx, cy - face_ry, cx + face_rx, cy + face_ry],
        fill=(220, 190, 170),
    )

    # "Eyes" - two dark circles
    eye_y = cy - face_ry // 3
    eye_offset = face_rx // 3
    eye_r = face_rx // 8
    draw.ellipse(
        [cx - eye_offset - eye_r, eye_y - eye_r,
         cx - eye_offset + eye_r, eye_y + eye_r],
        fill=(80, 60, 50),
    )
    draw.ellipse(
        [cx + eye_offset - eye_r, eye_y - eye_r,
         cx + eye_offset + eye_r, eye_y + eye_r],
        fill=(80, 60, 50),
    )

    # "Nose" - small triangle/line
    nose_top = cy - face_ry // 6
    nose_bottom = cy + face_ry // 6
    draw.line([(cx, nose_top), (cx, nose_bottom)], fill=(180, 150, 130), width=3)

    # "Mouth" - arc
    mouth_y = cy + face_ry // 3
    mouth_w = face_rx // 2
    draw.arc(
        [cx - mouth_w, mouth_y - 10, cx + mouth_w, mouth_y + 15],
        start=0, end=180, fill=(160, 100, 90), width=2,
    )

    return img


def test_preprocessor():
    """Test image preprocessing."""
    print("Testing Preprocessor...")
    prep = Preprocessor()
    img = create_test_image()

    # Test without background removal (faster, no rembg needed)
    processed, mask = prep.process(img, remove_background=False)
    assert processed.mode == "RGB"
    assert max(processed.size) <= 1024
    print(f"  Processed size: {processed.size}")
    print("  PASS")


def test_depth_estimator():
    """Test MiDaS depth estimation."""
    print("Testing DepthEstimator...")
    estimator = DepthEstimator()
    img = create_test_image(300, 400)

    depth = estimator.estimate(img, quality="preview", target_size=256)
    assert isinstance(depth, np.ndarray)
    assert depth.dtype == np.float32
    assert depth.min() >= 0.0
    assert depth.max() <= 1.0
    assert len(depth.shape) == 2
    print(f"  Depth map shape: {depth.shape}")
    print(f"  Depth range: [{depth.min():.3f}, {depth.max():.3f}]")
    print("  PASS")
    return depth


def test_face_detector():
    """Test face detection (may not find a face in synthetic image)."""
    print("Testing FaceDetector...")
    detector = FaceDetector()
    img = create_test_image()

    faces = detector.detect(img)
    print(f"  Faces detected: {len(faces)}")
    if faces:
        face = faces[0]
        print(f"  Bounding box: {face.bounding_box}")
        print(f"  Landmarks shape: {face.landmarks.shape}")
        print(f"  Region masks present: face_outline={face.region_masks.face_outline.sum():.0f}px")
    else:
        print("  (No face detected in synthetic image — this is expected)")
    print("  PASS")
    return faces


def test_refiner():
    """Test depth refinement with and without faces."""
    print("Testing DepthRefiner...")
    refiner = DepthRefiner()

    # Create a synthetic depth map
    h, w = 200, 150
    depth = np.random.rand(h, w).astype(np.float32) * 0.5 + 0.25
    # Add a "nose" bump
    cy, cx = h // 2, w // 2
    y, x = np.ogrid[:h, :w]
    nose_mask = ((x - cx) ** 2 + (y - cy) ** 2) < 400
    depth[nose_mask] += 0.2

    params = RefinementParams(eye_depth=0.3, detail_level=0.6)

    # Test without faces
    refined = refiner.refine(depth, [], params)
    assert refined.shape == depth.shape
    assert refined.dtype == np.float32
    assert refined.min() >= 0.0
    assert refined.max() <= 1.0
    print(f"  Refined (no face) range: [{refined.min():.3f}, {refined.max():.3f}]")
    print("  PASS")


def test_relief_mapper():
    """Test artistic relief curve mapping."""
    print("Testing ReliefMapper...")
    mapper = ReliefMapper()

    depth = np.random.rand(200, 150).astype(np.float32)

    for style in ReliefStyle:
        params = ReliefParams(style=style, max_depth_mm=6.0)
        relief = mapper.apply(depth, params)
        assert relief.shape == depth.shape
        assert relief.min() >= 0.0
        assert relief.max() <= params.max_depth_mm * 1.1  # small tolerance
        print(f"  {style.value}: range [{relief.min():.2f}, {relief.max():.2f}] mm")

    print("  PASS")


def test_mesh_generator():
    """Test mesh generation and STL export."""
    print("Testing MeshGenerator...")
    generator = MeshGenerator()

    # Create a simple dome relief
    h, w = 100, 120
    y, x = np.ogrid[:h, :w]
    cy, cx = h / 2, w / 2
    dist = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
    relief = np.maximum(0, 5.0 - dist * 0.1).astype(np.float32)

    mesh = generator.generate(
        relief,
        width_mm=100.0,
        base_thickness_mm=2.0,
        mesh_resolution=50,
    )

    assert mesh is not None
    assert len(mesh.vectors) > 0
    print(f"  Triangle count: {len(mesh.vectors)}")

    # Test STL export
    with tempfile.NamedTemporaryFile(suffix=".stl", delete=False) as f:
        tmppath = f.name
    try:
        generator.save_stl(mesh, tmppath)
        file_size = os.path.getsize(tmppath)
        print(f"  STL file size: {file_size:,} bytes")
        assert file_size > 0
    finally:
        os.unlink(tmppath)

    # Test bytes export
    stl_bytes = generator.to_bytes(mesh)
    assert len(stl_bytes) > 0
    print(f"  STL bytes: {len(stl_bytes):,}")
    print("  PASS")


def test_full_pipeline():
    """Test the complete pipeline end-to-end."""
    print("\nTesting Full Pipeline (end-to-end)...")
    pipeline = PortraitPipeline()
    img = create_test_image(400, 500)

    params = RefinementParams(eye_depth=0.3, detail_level=0.6)
    relief_params = ReliefParams(
        style=ReliefStyle.CLASSICAL,
        max_depth_mm=6.0,
        output_width_mm=150.0,
        vignette=True,
    )

    result = pipeline.process(
        img,
        quality="preview",
        refinement_params=params,
        relief_params=relief_params,
        remove_background=False,  # Skip rembg for speed in test
        mesh_resolution=100,
    )

    assert result.mesh is not None
    assert result.raw_depth_map is not None
    assert result.refined_depth_map is not None
    assert result.relief_map is not None
    print(f"  Faces detected: {len(result.faces_detected)}")
    print(f"  Depth map shape: {result.raw_depth_map.shape}")
    print(f"  Relief range: [{result.relief_map.min():.2f}, {result.relief_map.max():.2f}] mm")
    print(f"  Mesh triangles: {len(result.mesh.vectors)}")

    # Test STL export
    with tempfile.NamedTemporaryFile(suffix=".stl", delete=False) as f:
        tmppath = f.name
    try:
        result.save_stl(tmppath)
        file_size = os.path.getsize(tmppath)
        print(f"  STL file size: {file_size:,} bytes")
    finally:
        os.unlink(tmppath)

    # Test preview images
    depth_preview = result.depth_preview()
    relief_preview = result.relief_preview()
    assert depth_preview.mode == "L"
    assert relief_preview.mode == "L"
    print(f"  Depth preview: {depth_preview.size}")
    print(f"  Relief preview: {relief_preview.size}")

    print(f"\n  Timings:")
    for stage, t in result.timings.items():
        print(f"    {stage}: {t:.2f}s")

    print("  PASS")


if __name__ == "__main__":
    print("=" * 60)
    print("Portrait STL Generator — Integration Tests")
    print("=" * 60)
    print()

    test_preprocessor()
    print()
    test_depth_estimator()
    print()
    test_face_detector()
    print()
    test_refiner()
    print()
    test_relief_mapper()
    print()
    test_mesh_generator()
    print()
    test_full_pipeline()

    print()
    print("=" * 60)
    print("ALL TESTS PASSED")
    print("=" * 60)
