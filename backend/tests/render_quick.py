#!/usr/bin/env python3
"""Quick render all 4 portraits with shaded 3D preview."""

import sys, os, time
import numpy as np
from PIL import Image
from scipy import ndimage
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.pipeline.orchestrator import PortraitPipeline
from app.pipeline.refiner import RefinementParams
from app.pipeline.relief import ReliefParams, ReliefStyle


def render_shaded(depth_path, output_path, z_scale=4.0):
    """Render depth map with dramatic directional lighting to show relief."""
    img = Image.open(depth_path).convert("L")
    depth = np.array(img, dtype=np.float64) / 255.0

    # Light smooth for cleaner normals
    depth_smooth = ndimage.gaussian_filter(depth, sigma=0.8)

    dy, dx = np.gradient(depth_smooth)
    dx *= z_scale
    dy *= z_scale

    # Light from upper-left at a steep angle to cast shadows
    az, alt = np.radians(230), np.radians(30)
    lx = np.cos(alt) * np.cos(az)
    ly = np.cos(alt) * np.sin(az)
    lz = np.sin(alt)

    norm = np.sqrt(dx**2 + dy**2 + 1.0)
    nx, ny, nz = -dx/norm, -dy/norm, 1.0/norm
    shading = nx*lx + ny*ly + nz*lz
    shading = np.clip(shading, 0, 1)

    # Two-tone lighting: warm key + cool fill
    key = shading ** 0.9  # Slightly boost mid-tones

    # Fill light from opposite side (softer)
    az2, alt2 = np.radians(60), np.radians(50)
    lx2 = np.cos(alt2) * np.cos(az2)
    ly2 = np.cos(alt2) * np.sin(az2)
    lz2 = np.sin(alt2)
    fill = nx*lx2 + ny*ly2 + nz*lz2
    fill = np.clip(fill, 0, 1) * 0.25

    result = 0.12 + 0.7 * key + fill

    # Specular highlight
    reflect_z = 2 * nz * (nx*lx + ny*ly + nz*lz) - lz
    specular = np.clip(reflect_z, 0, 1) ** 32 * 0.2
    result = np.clip(result + specular, 0, 1)

    # Boost contrast
    result = np.clip((result - 0.3) * 1.4 + 0.3, 0, 1)

    # Background
    bg_mask = depth < 0.03
    result[bg_mask] = 0.04

    # Rich wood tone
    r = (result * 215 + (1-result) * 50).astype(np.uint8)
    g = (result * 180 + (1-result) * 35).astype(np.uint8)
    b = (result * 135 + (1-result) * 20).astype(np.uint8)
    r[bg_mask], g[bg_mask], b[bg_mask] = 30, 22, 14

    Image.fromarray(np.stack([r,g,b], axis=-1)).save(output_path)


pipeline = PortraitPipeline()
img_dir = "/home/user/1/backend/test_images"
out_dir = "/home/user/1/backend/test_output"

for fname in sorted(os.listdir(img_dir)):
    if not fname.endswith((".jpg", ".png")):
        continue

    name = Path(fname).stem
    print(f"\n{'='*50}")
    print(f"Processing: {name}")
    path = os.path.join(img_dir, fname)
    image = Image.open(path).convert("RGB")
    print(f"  Input: {image.size[0]}x{image.size[1]}")

    t0 = time.time()
    result = pipeline.process(
        image,
        quality="preview",
        remove_background=True,
        mesh_resolution=300,
    )
    elapsed = time.time() - t0
    print(f"  Faces: {len(result.faces_detected)}")
    print(f"  Time: {elapsed:.1f}s")

    # Save outputs
    result.depth_preview().save(os.path.join(out_dir, f"{name}_v3_depth.png"))
    result.relief_preview().save(os.path.join(out_dir, f"{name}_v3_relief.png"))
    result.save_stl(os.path.join(out_dir, f"{name}_v3.stl"))

    # Render shaded
    render_shaded(
        os.path.join(out_dir, f"{name}_v3_depth.png"),
        os.path.join(out_dir, f"{name}_v3_wood.png"),
        z_scale=12.0,
    )
    print(f"  Saved depth, relief, STL, and wood render")

    # Quick face stats
    if result.faces_detected:
        face = result.faces_detected[0]
        d = result.refined_depth_map
        fm = face.region_masks
        face_med = np.median(d[fm.face_outline > 0.5])
        nose_med = np.median(d[fm.nose > 0.5]) if fm.nose.sum() > 10 else 0
        print(f"  Face median: {face_med:.3f}")
        print(f"  Nose projection: {nose_med - face_med:.3f} above face")

print("\nDone!")
