#!/usr/bin/env python3
"""
Portrait quality testing and parameter tuning.

Runs the pipeline on real portrait photos and produces diagnostic output
for analyzing depth map quality, face detection accuracy, and relief results.
"""

import os
import sys
import json
import numpy as np
from PIL import Image
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.pipeline.orchestrator import PortraitPipeline
from app.pipeline.refiner import RefinementParams
from app.pipeline.relief import ReliefParams, ReliefStyle
from app.pipeline.face import FaceData


def analyze_depth_map(depth: np.ndarray, name: str) -> dict:
    """Analyze a depth map for potential quality issues."""
    issues = []
    stats = {
        "shape": depth.shape,
        "min": float(depth.min()),
        "max": float(depth.max()),
        "mean": float(depth.mean()),
        "std": float(depth.std()),
    }

    # Check for flat/dead areas (no depth variation)
    if stats["std"] < 0.05:
        issues.append("FLAT: Very low depth variation — image may appear flat")

    # Check for extreme values dominating
    pct_near_zero = float((depth < 0.05).sum() / depth.size)
    pct_near_one = float((depth > 0.95).sum() / depth.size)
    stats["pct_near_zero"] = pct_near_zero
    stats["pct_near_one"] = pct_near_one

    if pct_near_zero > 0.7:
        issues.append(f"BACKGROUND_HEAVY: {pct_near_zero:.0%} of pixels near zero (background)")
    if pct_near_one > 0.3:
        issues.append(f"FOREGROUND_HEAVY: {pct_near_one:.0%} of pixels near max depth")

    # Check depth histogram distribution
    hist, bins = np.histogram(depth, bins=20)
    hist_normalized = hist / hist.sum()
    stats["histogram"] = [float(h) for h in hist_normalized]

    # Check for bimodal distribution (common with good bg removal)
    # vs uniform (bad depth estimation)
    top_2_bins = np.sort(hist_normalized)[-2:]
    if top_2_bins[0] > 0.3 and top_2_bins[1] > 0.3:
        issues.append("BIMODAL: Strong bimodal depth — check if this is intentional (bg/fg separation)")

    stats["issues"] = issues
    return stats


def analyze_face_regions(face: FaceData, depth: np.ndarray, name: str) -> dict:
    """Analyze depth values within each facial region for creepy artifacts."""
    masks = face.region_masks
    results = {}

    regions = {
        "left_eye": masks.left_eye,
        "right_eye": masks.right_eye,
        "nose": masks.nose,
        "mouth": masks.mouth,
        "forehead": masks.forehead,
        "left_cheek": masks.left_cheek,
        "right_cheek": masks.right_cheek,
        "chin": masks.chin,
        "face_outline": masks.face_outline,
    }

    face_depth_values = depth[masks.face_outline > 0.5]
    if len(face_depth_values) == 0:
        return {"error": "No face pixels found"}

    face_median = float(np.median(face_depth_values))
    face_std = float(np.std(face_depth_values))

    issues = []

    for region_name, mask in regions.items():
        region_pixels = depth[mask > 0.5]
        if len(region_pixels) == 0:
            results[region_name] = {"pixels": 0, "note": "empty mask"}
            continue

        region_stats = {
            "pixels": int(len(region_pixels)),
            "min": float(region_pixels.min()),
            "max": float(region_pixels.max()),
            "mean": float(region_pixels.mean()),
            "median": float(np.median(region_pixels)),
            "std": float(region_pixels.std()),
            "range": float(region_pixels.max() - region_pixels.min()),
            "vs_face_median": float(np.median(region_pixels) - face_median),
        }
        results[region_name] = region_stats

        # Check for creepy-specific issues
        if "eye" in region_name:
            # Eyes should NOT be much deeper than face surface
            # Threshold scaled for wider face depth range (0.35-1.0)
            depth_below_face = face_median - region_stats["median"]
            if depth_below_face > 0.20:
                issues.append(
                    f"HOLLOW_EYES ({region_name}): Eyes are {depth_below_face:.3f} "
                    f"below face median — will look sunken/dead"
                )
            # Eyes should not have too much internal variation
            if region_stats["range"] > 0.5:
                issues.append(
                    f"EYE_VARIATION ({region_name}): High depth range {region_stats['range']:.3f} "
                    f"within eye region — may look unnatural"
                )

        if region_name == "nose":
            # Nose should be the highest point
            nose_above_face = region_stats["median"] - face_median
            if nose_above_face < 0.04:
                issues.append(
                    f"FLAT_NOSE: Nose barely projects above face ({nose_above_face:.3f}) "
                    f"— will look flat/featureless"
                )
            if region_stats["range"] > 0.55:
                issues.append(
                    f"SPIKY_NOSE: Very high depth range in nose ({region_stats['range']:.3f}) "
                    f"— may create sharp spike"
                )

        if region_name == "mouth":
            if region_stats["range"] > 0.4:
                issues.append(
                    f"MOUTH_DEPTH: High depth variation in mouth ({region_stats['range']:.3f}) "
                    f"— may look like grimace/snarl"
                )

    results["face_median"] = face_median
    results["face_std"] = face_std
    results["issues"] = issues
    return results


def run_portrait_test(
    image_path: str,
    output_dir: str,
    refinement_params: RefinementParams,
    relief_params: ReliefParams,
    pipeline: PortraitPipeline,
    label: str = "",
) -> dict:
    """Run a single portrait through the pipeline with full diagnostics."""
    name = Path(image_path).stem
    if label:
        name = f"{name}_{label}"

    print(f"\n{'='*60}")
    print(f"Processing: {name}")
    print(f"{'='*60}")

    image = Image.open(image_path).convert("RGB")
    print(f"  Input: {image.size[0]}x{image.size[1]}")

    result = pipeline.process(
        image,
        quality="preview",
        refinement_params=refinement_params,
        relief_params=relief_params,
        remove_background=True,
        mesh_resolution=256,
    )

    # Save outputs
    result.save_stl(os.path.join(output_dir, f"{name}.stl"))
    result.depth_preview().save(os.path.join(output_dir, f"{name}_depth.png"))
    result.relief_preview().save(os.path.join(output_dir, f"{name}_relief.png"))

    # Save raw depth as 16-bit PNG for detailed analysis
    raw_16bit = (result.raw_depth_map * 65535).astype(np.uint16)
    Image.fromarray(raw_16bit, mode="I;16").save(
        os.path.join(output_dir, f"{name}_raw_depth_16bit.png")
    )

    # Analyze results
    print(f"\n  Faces detected: {len(result.faces_detected)}")
    print(f"  Timings:")
    for stage, t in result.timings.items():
        print(f"    {stage}: {t:.2f}s")

    # Depth map analysis
    print(f"\n  --- Raw Depth Map Analysis ---")
    raw_stats = analyze_depth_map(result.raw_depth_map, f"{name}_raw")
    print(f"  Range: [{raw_stats['min']:.3f}, {raw_stats['max']:.3f}]")
    print(f"  Mean: {raw_stats['mean']:.3f}, Std: {raw_stats['std']:.3f}")
    print(f"  Background (near 0): {raw_stats['pct_near_zero']:.1%}")
    print(f"  Foreground (near 1): {raw_stats['pct_near_one']:.1%}")
    for issue in raw_stats.get("issues", []):
        print(f"  ⚠ {issue}")

    print(f"\n  --- Refined Depth Map Analysis ---")
    refined_stats = analyze_depth_map(result.refined_depth_map, f"{name}_refined")
    print(f"  Range: [{refined_stats['min']:.3f}, {refined_stats['max']:.3f}]")
    print(f"  Mean: {refined_stats['mean']:.3f}, Std: {refined_stats['std']:.3f}")
    for issue in refined_stats.get("issues", []):
        print(f"  ⚠ {issue}")

    # Face region analysis
    face_analysis = {}
    for i, face in enumerate(result.faces_detected):
        print(f"\n  --- Face {i+1} Region Analysis (REFINED depth) ---")
        fa = analyze_face_regions(face, result.refined_depth_map, name)
        face_analysis[f"face_{i}"] = fa

        if "error" in fa:
            print(f"  Error: {fa['error']}")
            continue

        print(f"  Face median depth: {fa['face_median']:.3f}")
        for region in ["left_eye", "right_eye", "nose", "mouth", "forehead"]:
            if region in fa and "median" in fa[region]:
                r = fa[region]
                delta = r["vs_face_median"]
                direction = "above" if delta > 0 else "below"
                print(f"  {region:>12}: median={r['median']:.3f}, "
                      f"range={r['range']:.3f}, "
                      f"{abs(delta):.3f} {direction} face")

        if fa.get("issues"):
            print(f"\n  ISSUES FOUND:")
            for issue in fa["issues"]:
                print(f"    ⚠ {issue}")
        else:
            print(f"\n  ✓ No creepy artifacts detected")

    # Relief analysis
    print(f"\n  --- Relief Map ---")
    print(f"  Physical range: [{result.relief_map.min():.2f}, {result.relief_map.max():.2f}] mm")

    # STL analysis
    stl_path = os.path.join(output_dir, f"{name}.stl")
    stl_size = os.path.getsize(stl_path)
    print(f"  STL: {len(result.mesh.vectors)} triangles, {stl_size:,} bytes")

    return {
        "name": name,
        "faces": len(result.faces_detected),
        "raw_stats": raw_stats,
        "refined_stats": refined_stats,
        "face_analysis": face_analysis,
        "timings": result.timings,
        "relief_range_mm": [float(result.relief_map.min()), float(result.relief_map.max())],
        "stl_triangles": len(result.mesh.vectors),
    }


def main():
    img_dir = "/home/user/1/backend/test_images"
    out_dir = "/home/user/1/backend/test_output"
    os.makedirs(out_dir, exist_ok=True)

    images = sorted([
        os.path.join(img_dir, f)
        for f in os.listdir(img_dir)
        if f.endswith((".jpg", ".jpeg", ".png"))
    ])

    if not images:
        print("No test images found in", img_dir)
        return

    print(f"Found {len(images)} test images")

    pipeline = PortraitPipeline()

    # Default parameters
    default_refinement = RefinementParams()
    default_relief = ReliefParams(style=ReliefStyle.CLASSICAL)

    all_results = []

    for img_path in images:
        result = run_portrait_test(
            img_path, out_dir,
            default_refinement, default_relief,
            pipeline, label="default"
        )
        all_results.append(result)

    # Summary
    print(f"\n\n{'='*60}")
    print(f"SUMMARY — Default Parameters")
    print(f"{'='*60}")

    all_issues = []
    for r in all_results:
        print(f"\n{r['name']}:")
        print(f"  Faces: {r['faces']}, Triangles: {r['stl_triangles']}")
        print(f"  Relief: {r['relief_range_mm'][0]:.2f} - {r['relief_range_mm'][1]:.2f} mm")
        for face_key, fa in r.get("face_analysis", {}).items():
            issues = fa.get("issues", [])
            if issues:
                print(f"  Issues ({face_key}):")
                for issue in issues:
                    print(f"    ⚠ {issue}")
                    all_issues.append((r["name"], issue))
            else:
                print(f"  ✓ {face_key}: No issues")

    if all_issues:
        print(f"\n\n{'='*60}")
        print(f"ALL ISSUES TO FIX ({len(all_issues)} total)")
        print(f"{'='*60}")
        for name, issue in all_issues:
            print(f"  [{name}] {issue}")
    else:
        print(f"\n✓ No issues found across all portraits!")

    # Save full results as JSON
    results_path = os.path.join(out_dir, "results.json")
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nDetailed results saved to: {results_path}")


if __name__ == "__main__":
    main()
