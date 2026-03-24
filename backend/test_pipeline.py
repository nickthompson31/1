#!/usr/bin/env python3
"""
Test script for the portrait STL pipeline.

Usage:
    python test_pipeline.py path/to/portrait.jpg [--quality preview|final] [--output output.stl]

This will process the image and save:
    - output.stl          — The 3D relief mesh
    - output_depth.png    — The refined depth map
    - output_relief.png   — The relief height map
"""

import argparse
import sys
import time
from pathlib import Path

from PIL import Image

from app.pipeline.orchestrator import PortraitPipeline
from app.pipeline.refiner import RefinementParams
from app.pipeline.relief import ReliefParams, ReliefStyle


def main():
    parser = argparse.ArgumentParser(
        description="Convert a portrait photo to a carve-ready STL file"
    )
    parser.add_argument("image", help="Path to portrait image (JPG, PNG)")
    parser.add_argument(
        "--quality",
        choices=["preview", "final"],
        default="preview",
        help="Processing quality (default: preview)",
    )
    parser.add_argument(
        "--output", "-o",
        default="output.stl",
        help="Output STL file path (default: output.stl)",
    )
    parser.add_argument(
        "--style",
        choices=["classical", "roman", "dramatic", "subtle", "coin"],
        default="roman",
        help="Relief style preset (default: roman)",
    )
    parser.add_argument(
        "--depth", type=float, default=8.0,
        help="Maximum relief depth in mm (default: 8.0)",
    )
    parser.add_argument(
        "--width", type=float, default=150.0,
        help="Output width in mm (default: 150.0 / ~6 inches)",
    )
    parser.add_argument(
        "--eye-depth", type=float, default=0.6,
        help="Eye socket depth 0-1 (default: 0.6)",
    )
    parser.add_argument(
        "--detail", type=float, default=0.25,
        help="Detail preservation 0-1, lower=smoother classical look (default: 0.25)",
    )
    parser.add_argument(
        "--no-bg-remove", action="store_true",
        help="Skip background removal",
    )
    parser.add_argument(
        "--no-vignette", action="store_true",
        help="Disable edge vignette",
    )
    parser.add_argument(
        "--mesh-res", type=int, default=None,
        help="Mesh resolution (max grid dimension). None=auto",
    )

    args = parser.parse_args()

    # Load image
    image_path = Path(args.image)
    if not image_path.exists():
        print(f"Error: Image not found: {image_path}")
        sys.exit(1)

    print(f"Loading image: {image_path}")
    image = Image.open(image_path).convert("RGB")
    print(f"  Image size: {image.size[0]}x{image.size[1]}")

    # Configure parameters
    refinement_params = RefinementParams(
        eye_depth=args.eye_depth,
        detail_level=args.detail,
    )

    relief_params = ReliefParams(
        max_depth_mm=args.depth,
        style=ReliefStyle(args.style),
        output_width_mm=args.width,
        vignette=not args.no_vignette,
    )

    # Process
    print(f"\nProcessing with quality={args.quality}, style={args.style}...")
    pipeline = PortraitPipeline()

    t_start = time.time()
    result = pipeline.process(
        image,
        quality=args.quality,
        refinement_params=refinement_params,
        relief_params=relief_params,
        remove_background=not args.no_bg_remove,
        mesh_resolution=args.mesh_res,
    )
    total_time = time.time() - t_start

    # Print results
    print(f"\nResults:")
    print(f"  Faces detected: {len(result.faces_detected)}")
    print(f"  Timings:")
    for stage, duration in result.timings.items():
        print(f"    {stage}: {duration:.2f}s")
    print(f"  Total: {total_time:.2f}s")

    # Save outputs
    output_path = Path(args.output)
    stem = output_path.stem

    # STL
    result.save_stl(str(output_path))
    print(f"\n  STL saved: {output_path}")

    # Depth map preview
    depth_path = output_path.parent / f"{stem}_depth.png"
    result.depth_preview().save(str(depth_path))
    print(f"  Depth map: {depth_path}")

    # Relief map preview
    relief_path = output_path.parent / f"{stem}_relief.png"
    result.relief_preview().save(str(relief_path))
    print(f"  Relief map: {relief_path}")

    print(f"\nDone! Open {output_path} in your 3D viewer or CAM software.")


if __name__ == "__main__":
    main()
