"""
API routes for the portrait STL generator.

Endpoints:
- POST /api/process          — Full pipeline: upload image → get STL
- POST /api/preview          — Quick preview: upload image → get depth preview
- POST /api/depth-map        — Get just the depth map image
- GET  /api/health           — Health check
"""

import io
import base64
import traceback

from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse
from PIL import Image

from ..pipeline.orchestrator import PortraitPipeline
from ..pipeline.refiner import RefinementParams
from ..pipeline.relief import ReliefParams, ReliefStyle

router = APIRouter(prefix="/api")

# Singleton pipeline instance (models loaded lazily)
_pipeline: PortraitPipeline | None = None


def get_pipeline() -> PortraitPipeline:
    global _pipeline
    if _pipeline is None:
        _pipeline = PortraitPipeline()
    return _pipeline


def _parse_refinement_params(
    feature_strength: float = 0.8,
    eye_depth: float = 0.4,
    nose_projection: float = 0.85,
    mouth_depth: float = 0.5,
    forehead_roundness: float = 0.6,
    cheek_volume: float = 0.6,
    jaw_definition: float = 0.6,
    detail_level: float = 0.5,
    smoothing: float = 0.3,
) -> RefinementParams:
    return RefinementParams(
        feature_strength=feature_strength,
        eye_depth=eye_depth,
        nose_projection=nose_projection,
        mouth_depth=mouth_depth,
        forehead_roundness=forehead_roundness,
        cheek_volume=cheek_volume,
        jaw_definition=jaw_definition,
        detail_level=detail_level,
        smoothing=smoothing,
    )


def _parse_relief_params(
    max_depth_mm: float = 6.0,
    style: str = "classical",
    output_width_mm: float = 150.0,
    base_thickness_mm: float = 2.0,
    vignette: bool = True,
    vignette_strength: float = 0.8,
    invert: bool = False,
) -> ReliefParams:
    return ReliefParams(
        max_depth_mm=max_depth_mm,
        style=ReliefStyle(style),
        output_width_mm=output_width_mm,
        base_thickness_mm=base_thickness_mm,
        vignette=vignette,
        vignette_strength=vignette_strength,
        invert=invert,
    )


@router.get("/health")
async def health_check():
    return {"status": "ok", "service": "portrait-stl-generator"}


@router.post("/preview")
async def generate_preview(
    file: UploadFile = File(...),
    # Refinement params
    feature_strength: float = Form(0.8),
    eye_depth: float = Form(0.4),
    nose_projection: float = Form(0.85),
    mouth_depth: float = Form(0.5),
    detail_level: float = Form(0.5),
    smoothing: float = Form(0.3),
    # Relief params
    max_depth_mm: float = Form(6.0),
    style: str = Form("classical"),
    vignette: bool = Form(True),
    remove_background: bool = Form(True),
):
    """
    Generate a quick preview — returns depth map image and timing info.
    Uses the fast model for rapid iteration.
    """
    try:
        image = Image.open(io.BytesIO(await file.read())).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image file")

    pipeline = get_pipeline()

    refinement_params = _parse_refinement_params(
        feature_strength=feature_strength,
        eye_depth=eye_depth,
        nose_projection=nose_projection,
        mouth_depth=mouth_depth,
        detail_level=detail_level,
        smoothing=smoothing,
    )

    relief_params = _parse_relief_params(
        max_depth_mm=max_depth_mm,
        style=style,
        vignette=vignette,
    )

    try:
        result = pipeline.process(
            image,
            quality="preview",
            refinement_params=refinement_params,
            relief_params=relief_params,
            remove_background=remove_background,
            mesh_resolution=200,
        )
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")

    # Return depth preview as base64 image + metadata
    depth_img = result.depth_preview()
    relief_img = result.relief_preview()

    depth_buf = io.BytesIO()
    depth_img.save(depth_buf, format="PNG")
    depth_b64 = base64.b64encode(depth_buf.getvalue()).decode()

    relief_buf = io.BytesIO()
    relief_img.save(relief_buf, format="PNG")
    relief_b64 = base64.b64encode(relief_buf.getvalue()).decode()

    return JSONResponse({
        "depth_preview": f"data:image/png;base64,{depth_b64}",
        "relief_preview": f"data:image/png;base64,{relief_b64}",
        "faces_detected": len(result.faces_detected),
        "timings": result.timings,
        "image_size": {
            "width": result.preprocessed_image.size[0],
            "height": result.preprocessed_image.size[1],
        },
    })


@router.post("/process")
async def generate_stl(
    file: UploadFile = File(...),
    quality: str = Form("final"),
    # Refinement params
    feature_strength: float = Form(0.8),
    eye_depth: float = Form(0.4),
    nose_projection: float = Form(0.85),
    mouth_depth: float = Form(0.5),
    forehead_roundness: float = Form(0.6),
    cheek_volume: float = Form(0.6),
    jaw_definition: float = Form(0.6),
    detail_level: float = Form(0.5),
    smoothing: float = Form(0.3),
    # Relief params
    max_depth_mm: float = Form(6.0),
    style: str = Form("classical"),
    output_width_mm: float = Form(150.0),
    base_thickness_mm: float = Form(2.0),
    vignette: bool = Form(True),
    vignette_strength: float = Form(0.7),
    invert: bool = Form(False),
    remove_background: bool = Form(True),
    # Mesh params
    mesh_resolution: int = Form(512),
):
    """
    Generate the final STL file. Returns the STL as a binary download.
    """
    try:
        image = Image.open(io.BytesIO(await file.read())).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image file")

    pipeline = get_pipeline()

    refinement_params = _parse_refinement_params(
        feature_strength=feature_strength,
        eye_depth=eye_depth,
        nose_projection=nose_projection,
        mouth_depth=mouth_depth,
        forehead_roundness=forehead_roundness,
        cheek_volume=cheek_volume,
        jaw_definition=jaw_definition,
        detail_level=detail_level,
        smoothing=smoothing,
    )

    relief_params = _parse_relief_params(
        max_depth_mm=max_depth_mm,
        style=style,
        output_width_mm=output_width_mm,
        base_thickness_mm=base_thickness_mm,
        vignette=vignette,
        vignette_strength=vignette_strength,
        invert=invert,
    )

    try:
        result = pipeline.process(
            image,
            quality=quality,
            refinement_params=refinement_params,
            relief_params=relief_params,
            remove_background=remove_background,
            mesh_resolution=mesh_resolution,
        )
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")

    # Return STL as binary stream
    stl_data = result.stl_bytes()

    return StreamingResponse(
        io.BytesIO(stl_data),
        media_type="application/octet-stream",
        headers={
            "Content-Disposition": "attachment; filename=portrait_relief.stl",
            "X-Faces-Detected": str(len(result.faces_detected)),
            "X-Processing-Time": f"{result.timings.get('total', 0):.2f}s",
        },
    )


@router.post("/depth-map")
async def get_depth_map(
    file: UploadFile = File(...),
    quality: str = Form("preview"),
    remove_background: bool = Form(True),
    # Refinement params
    feature_strength: float = Form(0.8),
    eye_depth: float = Form(0.4),
    nose_projection: float = Form(0.85),
    mouth_depth: float = Form(0.5),
    detail_level: float = Form(0.5),
    smoothing: float = Form(0.3),
):
    """Return the refined depth map as a PNG image."""
    try:
        image = Image.open(io.BytesIO(await file.read())).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image file")

    pipeline = get_pipeline()

    refinement_params = _parse_refinement_params(
        feature_strength=feature_strength,
        eye_depth=eye_depth,
        nose_projection=nose_projection,
        mouth_depth=mouth_depth,
        detail_level=detail_level,
        smoothing=smoothing,
    )

    try:
        result = pipeline.process(
            image,
            quality=quality,
            refinement_params=refinement_params,
            remove_background=remove_background,
        )
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")

    depth_img = result.depth_preview()
    buf = io.BytesIO()
    depth_img.save(buf, format="PNG")
    buf.seek(0)

    return StreamingResponse(buf, media_type="image/png")
