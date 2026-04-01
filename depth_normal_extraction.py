"""
Depth & Surface Normal Extraction — Proof of Concept

First-principles pipeline foundation:
1. Take a 2D image as input
2. Extract raw Z-depth using a state-of-the-art AI depth model
3. Calculate surface normal vectors (gradients) from that depth data

Usage:
    python depth_normal_extraction.py [image_path]

If no image is provided, a synthetic test scene is generated.
"""

import sys
import os
import torch
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# 1. Load an input image
image_path = sys.argv[1] if len(sys.argv) > 1 else "sample_input.png"

if not os.path.exists(image_path):
    print(f"Generating synthetic test image at {image_path}...")
    w, h = 640, 480
    arr = np.zeros((h, w, 3), dtype=np.uint8)
    # Sky gradient
    for y in range(h // 2):
        t = y / (h // 2)
        arr[y, :] = [int(100 + 80 * t), int(150 + 50 * t), int(220 - 20 * t)]
    # Ground gradient
    for y in range(h // 2, h):
        t = (y - h // 2) / (h // 2)
        arr[y, :] = [int(80 + 40 * t), int(120 + 30 * t), int(60 + 20 * t)]
    # Building
    arr[100:350, 200:440] = [160, 140, 120]
    for wy in range(120, 340, 50):
        for wx in range(220, 430, 60):
            arr[wy:wy+30, wx:wx+35] = [80, 100, 130]
    # Foreground sphere
    yy, xx = np.ogrid[:h, :w]
    arr[((xx - 500)**2 + (yy - 350)**2) < 60**2] = [180, 60, 60]
    Image.fromarray(arr).save(image_path)

print(f"Loading image: {image_path}")
img = Image.open(image_path).convert("RGB")
print(f"  Image size: {img.size[0]}x{img.size[1]}")

# 2. Initialize the Depth Estimation Model via HuggingFace Transformers
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Loading depth estimation model onto {device}...")

from transformers import AutoImageProcessor, AutoModelForDepthEstimation

model_name = "Intel/zoedepth-nyu-kitti"
processor = AutoImageProcessor.from_pretrained(model_name)
model = AutoModelForDepthEstimation.from_pretrained(model_name).to(device)
model.eval()

# 3. Extract the Z-Depth Map
print("Extracting volumetric depth...")
inputs = processor(images=img, return_tensors="pt").to(device)

with torch.no_grad():
    outputs = model(**inputs)
    predicted_depth = outputs.predicted_depth

# Interpolate to original image size
predicted_depth = torch.nn.functional.interpolate(
    predicted_depth.unsqueeze(1),
    size=img.size[::-1],  # (height, width)
    mode="bicubic",
    align_corners=False,
).squeeze()

depth_numpy = predicted_depth.cpu().numpy()

# Normalize the depth to a 0.0 - 1.0 range for math operations
depth_min, depth_max = depth_numpy.min(), depth_numpy.max()
depth_normalized = (depth_numpy - depth_min) / (depth_max - depth_min)
print(f"  Depth range: {depth_min:.3f} - {depth_max:.3f}")
print(f"  Depth map shape: {depth_numpy.shape}")

# 4. Compute Surface Normals using First Principles (Gradients)
print("Calculating surface normals from the depth gradient...")
# We use NumPy to find the rate of change (slope) along the X and Y axes
dzdx = np.gradient(depth_normalized, axis=1)
dzdy = np.gradient(depth_normalized, axis=0)

# Construct the normal vectors [-dz/dx, -dz/dy, 1]
# A multiplier can be added here later to exaggerate the relief depth
normal_map = np.dstack((-dzdx, -dzdy, np.ones_like(depth_normalized)))

# Normalize the vectors so their length equals 1
norm = np.linalg.norm(normal_map, axis=2, keepdims=True)
normal_map_normalized = normal_map / norm

# Convert vector math (-1 to 1) into RGB color space (0 to 1) for visualization
normal_vis = (normal_map_normalized + 1.0) / 2.0

# 5. Render the outputs
print("Rendering visualization...")
fig, axs = plt.subplots(1, 3, figsize=(18, 6))

axs[0].imshow(img)
axs[0].set_title("1. Original Image")
axs[0].axis('off')

axs[1].imshow(depth_normalized, cmap='inferno')
axs[1].set_title("2. AI Extracted Global Depth")
axs[1].axis('off')

axs[2].imshow(normal_vis)
axs[2].set_title("3. Mathematically Derived Surface Normals")
axs[2].axis('off')

plt.tight_layout()

output_path = "depth_normal_output.png"
plt.savefig(output_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"\nDone! Output saved to: {output_path}")

# Also save the raw data for the next pipeline stage
np.save("depth_normalized.npy", depth_normalized)
np.save("normal_map.npy", normal_map_normalized)
print("Raw data saved: depth_normalized.npy, normal_map.npy")
