#!/usr/bin/env python3
"""Render all portraits with high quality 3D preview — no 8-bit banding."""

import sys, os, time
import numpy as np
from PIL import Image
from scipy import ndimage
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.pipeline.orchestrator import PortraitPipeline
from app.pipeline.refiner import RefinementParams
from app.pipeline.relief import ReliefParams, ReliefStyle


def render_from_float32(depth_f32, z_exag=0.40):
    """
    Render 3D relief directly from float32 depth — eliminates banding.
    Returns PIL Image.
    """
    depth = depth_f32.astype(np.float64)
    h, w = depth.shape

    # Smooth very slightly for clean normals
    depth_smooth = ndimage.gaussian_filter(depth, sigma=0.6)

    Z = depth_smooth * w * z_exag

    # Sobel for smooth surface normals
    dx = ndimage.sobel(Z, axis=1) / 8.0
    dy = ndimage.sobel(Z, axis=0) / 8.0

    norm = np.sqrt(dx**2 + dy**2 + 1.0)
    nx, ny, nz = -dx / norm, -dy / norm, 1.0 / norm

    # === THREE-POINT LIGHTING ===
    # Key light: upper-left, dramatic angle
    az1, alt1 = np.radians(225), np.radians(30)
    kd = np.array([np.cos(alt1)*np.cos(az1), np.cos(alt1)*np.sin(az1), np.sin(alt1)])
    key = np.clip(nx*kd[0] + ny*kd[1] + nz*kd[2], 0, 1) ** 0.9

    # Fill light: right side, softer and higher
    az2, alt2 = np.radians(340), np.radians(55)
    fd = np.array([np.cos(alt2)*np.cos(az2), np.cos(alt2)*np.sin(az2), np.sin(alt2)])
    fill = np.clip(nx*fd[0] + ny*fd[1] + nz*fd[2], 0, 1)

    # Rim/back light: from behind-below for edge definition
    az3, alt3 = np.radians(30), np.radians(8)
    rd = np.array([np.cos(alt3)*np.cos(az3), np.cos(alt3)*np.sin(az3), np.sin(alt3)])
    rim = np.clip(nx*rd[0] + ny*rd[1] + nz*rd[2], 0, 1)

    # Specular highlight
    view = np.array([0, 0, 1.0])
    hv = kd + view
    hv /= np.linalg.norm(hv)
    spec = np.clip(nx*hv[0] + ny*hv[1] + nz*hv[2], 0, 1) ** 60 * 0.10

    # Ambient occlusion — darkens crevices (eyes, nostrils, lip line)
    depth_blur = ndimage.gaussian_filter(depth, sigma=12)
    ao = np.clip(1.0 - (depth_blur - depth) * 6.0, 0.35, 1.0)

    # Combine
    result = (0.06 + 0.62 * key + 0.16 * fill + 0.08 * rim + spec) * ao
    result = np.clip(result, 0, 1)

    # Background mask
    bg = depth < 0.025

    # Warm walnut wood tones
    r = np.clip(result * 210 + (1 - result) * 48, 0, 255).astype(np.uint8)
    g = np.clip(result * 172 + (1 - result) * 36, 0, 255).astype(np.uint8)
    b = np.clip(result * 128 + (1 - result) * 22, 0, 255).astype(np.uint8)
    r[bg], g[bg], b[bg] = 22, 16, 10

    return Image.fromarray(np.stack([r, g, b], axis=-1))


def make_comparison(src_img, render_img, output_path):
    """Side-by-side: source left, render right."""
    target_h = 1000

    src_w = int(src_img.size[0] * target_h / src_img.size[1])
    src = src_img.resize((src_w, target_h), Image.LANCZOS)

    render_w = int(render_img.size[0] * target_h / render_img.size[1])
    render = render_img.resize((render_w, target_h), Image.LANCZOS)

    gap = 12
    total_w = src_w + gap + render_w
    canvas = Image.new("RGB", (total_w, target_h), (18, 13, 8))
    canvas.paste(src, (0, 0))
    canvas.paste(render, (src_w + gap, 0))
    canvas.save(output_path, quality=95)
    print(f"  Saved: {output_path} ({total_w}x{target_h})")


# ===== MAIN =====
pipeline = PortraitPipeline()
img_dir = "/home/user/1/backend/test_images"
out_dir = "/home/user/1/backend/test_output"
results_dir = "/home/user/1/results"
os.makedirs(results_dir, exist_ok=True)

for fname in sorted(os.listdir(img_dir)):
    if not fname.endswith((".jpg", ".png")):
        continue

    name = Path(fname).stem
    print(f"\n{'=' * 50}")
    print(f"Processing: {name}")

    image = Image.open(os.path.join(img_dir, fname)).convert("RGB")
    print(f"  Input: {image.size[0]}x{image.size[1]}")

    t0 = time.time()
    result = pipeline.process(
        image,
        quality="preview",
        remove_background=True,
        mesh_resolution=400,
    )
    elapsed = time.time() - t0
    print(f"  Faces: {len(result.faces_detected)}")
    print(f"  Time: {elapsed:.1f}s")
    print(f"  Depth shape: {result.refined_depth_map.shape}")

    # Render directly from float32 depth — no 8-bit banding!
    render = render_from_float32(result.refined_depth_map, z_exag=0.40)

    # Save comparison
    comparison_path = os.path.join(results_dir, f"{name}_final.png")
    make_comparison(image, render, comparison_path)

    # Also save the STL
    stl_path = os.path.join(out_dir, f"{name}_v4.stl")
    result.save_stl(stl_path)
    stl_size = os.path.getsize(stl_path)
    print(f"  STL: {stl_size:,} bytes, {len(result.mesh.vectors)} triangles")

    # Face stats
    if result.faces_detected:
        d = result.refined_depth_map
        fm = result.faces_detected[0].region_masks
        face_med = np.median(d[fm.face_outline > 0.5])
        nose_med = np.median(d[fm.nose > 0.5]) if fm.nose.sum() > 10 else 0
        print(f"  Nose projection: {nose_med - face_med:.3f} above face")

print("\nDone!")
