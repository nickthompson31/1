"""
Depth & Surface Normal Extraction — Web UI

Interactive Gradio interface for the first-principles depth/normal pipeline.
Upload an image, extract Z-depth via ZoeDepth, and compute surface normals.

Usage:
    python app.py
    # Opens at http://localhost:7860
"""

import torch
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import gradio as gr
from transformers import AutoImageProcessor, AutoModelForDepthEstimation
import io

# Load model once at startup
print("Loading ZoeDepth model...")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_name = "Intel/zoedepth-nyu-kitti"
processor = AutoImageProcessor.from_pretrained(model_name)
model = AutoModelForDepthEstimation.from_pretrained(model_name).to(device)
model.eval()
print(f"Model loaded on {device}")


def extract_depth_and_normals(input_image, normal_strength):
    """Run the full extraction pipeline on an uploaded image."""
    if input_image is None:
        raise gr.Error("Please upload an image.")

    img = Image.fromarray(input_image).convert("RGB")
    w, h = img.size

    # Extract depth
    inputs = processor(images=img, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)
        predicted_depth = outputs.predicted_depth

    predicted_depth = torch.nn.functional.interpolate(
        predicted_depth.unsqueeze(1),
        size=(h, w),
        mode="bicubic",
        align_corners=False,
    ).squeeze()

    depth_numpy = predicted_depth.cpu().numpy()
    depth_min, depth_max = depth_numpy.min(), depth_numpy.max()
    depth_normalized = (depth_numpy - depth_min) / (depth_max - depth_min)

    # Compute surface normals from gradients
    dzdx = np.gradient(depth_normalized, axis=1)
    dzdy = np.gradient(depth_normalized, axis=0)

    # Apply strength multiplier to exaggerate relief
    normal_map = np.dstack((
        -dzdx * normal_strength,
        -dzdy * normal_strength,
        np.ones_like(depth_normalized),
    ))
    norm = np.linalg.norm(normal_map, axis=2, keepdims=True)
    normal_map_normalized = normal_map / norm

    # Convert to RGB for display
    normal_vis = ((normal_map_normalized + 1.0) / 2.0 * 255).astype(np.uint8)

    # Render depth as inferno colormap
    cmap = plt.get_cmap("inferno")
    depth_colored = (cmap(depth_normalized)[:, :, :3] * 255).astype(np.uint8)

    info = (
        f"Depth range: {depth_min:.3f} – {depth_max:.3f} m\n"
        f"Resolution: {w} x {h}\n"
        f"Device: {device}"
    )

    return depth_colored, normal_vis, info


# Build the Gradio UI
with gr.Blocks(title="Depth & Normal Extraction") as demo:
    gr.Markdown("# Depth & Surface Normal Extraction")
    gr.Markdown(
        "Upload a 2D image to extract AI depth and mathematically derived "
        "surface normals using ZoeDepth + gradient computation."
    )

    with gr.Row():
        with gr.Column(scale=1):
            input_image = gr.Image(label="Input Image", type="numpy")
            normal_strength = gr.Slider(
                minimum=1.0, maximum=20.0, value=5.0, step=0.5,
                label="Normal Strength",
                info="Multiplier to exaggerate surface relief detail",
            )
            run_btn = gr.Button("Extract", variant="primary", size="lg")

        with gr.Column(scale=2):
            with gr.Row():
                depth_output = gr.Image(label="AI Extracted Global Depth")
                normal_output = gr.Image(label="Derived Surface Normals")
            info_output = gr.Textbox(label="Info", lines=3)

    run_btn.click(
        fn=extract_depth_and_normals,
        inputs=[input_image, normal_strength],
        outputs=[depth_output, normal_output, info_output],
    )

demo.launch(server_name="0.0.0.0", server_port=7860, theme=gr.themes.Soft())
