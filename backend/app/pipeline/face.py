"""
Face detection and landmark extraction using MediaPipe Tasks API.

Provides 478-point face mesh landmarks for precise facial region segmentation.
This enables region-specific depth treatment — the core of our "anti-creepy" pipeline.
"""

import os
import numpy as np
from dataclasses import dataclass
from pathlib import Path
from PIL import Image

import cv2
import mediapipe as mp
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import (
    FaceLandmarker,
    FaceLandmarkerOptions,
    RunningMode,
)


@dataclass
class FaceRegionMasks:
    """Binary masks for different facial regions, all same shape as input image."""

    face_outline: np.ndarray  # Full face boundary
    left_eye: np.ndarray
    right_eye: np.ndarray
    nose: np.ndarray
    mouth: np.ndarray
    forehead: np.ndarray
    left_cheek: np.ndarray
    right_cheek: np.ndarray
    chin: np.ndarray
    combined: np.ndarray  # Union of all face regions

    @property
    def eyes(self) -> np.ndarray:
        return np.maximum(self.left_eye, self.right_eye)


@dataclass
class FaceData:
    """Complete face analysis results."""

    landmarks: np.ndarray  # (N, 2) normalized landmark coords
    pixel_landmarks: np.ndarray  # (N, 2) pixel coordinates
    region_masks: FaceRegionMasks
    bounding_box: tuple[int, int, int, int]  # x, y, w, h
    confidence: float


# MediaPipe Face Mesh landmark index groups (478-point mesh)
# These indices are stable across MediaPipe versions
LANDMARK_REGIONS = {
    "face_outline": [
        10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288,
        397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136,
        172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109,
    ],
    "left_eye_wide": [
        # Wider region around left eye (for depth blending)
        336, 296, 334, 293, 300, 383, 372, 345, 352, 376, 433, 416,
        434, 430, 431, 262, 414, 286, 258, 368, 336,
    ],
    "right_eye_wide": [
        # Wider region around right eye
        107, 66, 105, 63, 70, 156, 143, 116, 123, 147, 213, 192,
        214, 210, 211, 32, 190, 56, 28, 139, 107,
    ],
    "nose": [
        # Nose bridge to tip
        168, 6, 197, 195, 5, 4, 1, 19, 94, 2, 164,
        0, 267, 269, 270, 409, 291, 305, 289,
        392, 309, 459, 458, 457, 274, 354, 19,
        75, 60, 20, 238, 239, 241, 125, 44, 237,
        79, 218, 237, 44, 125, 141, 235, 168,
    ],
    "mouth_tight": [
        # Just the lips area
        61, 146, 91, 181, 84, 17, 314, 405, 321, 375,
        291, 308, 324, 318, 402, 317, 14, 87, 178, 88,
        95, 78, 191, 80, 81, 82, 13, 312, 311, 310, 415,
    ],
}

# Default model path — relative to this file's parent (backend/)
_DEFAULT_MODEL_PATH = (
    Path(__file__).resolve().parent.parent.parent
    / "models"
    / "face_landmarker_v2_with_blendshapes.task"
)


class FaceDetector:
    """Detects faces and extracts landmark-based region masks."""

    def __init__(self, model_path: str | None = None):
        self._landmarker: FaceLandmarker | None = None
        self._model_path = model_path or str(_DEFAULT_MODEL_PATH)

    def _init_mediapipe(self):
        if self._landmarker is not None:
            return

        if not os.path.exists(self._model_path):
            raise FileNotFoundError(
                f"Face landmarker model not found at: {self._model_path}\n"
                "Download it from: https://storage.googleapis.com/mediapipe-models/"
                "face_landmarker/face_landmarker/float16/1/face_landmarker.task"
            )

        options = FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=self._model_path),
            running_mode=RunningMode.IMAGE,
            num_faces=4,
            min_face_detection_confidence=0.5,
            min_face_presence_confidence=0.5,
            min_tracking_confidence=0.5,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False,
        )
        self._landmarker = FaceLandmarker.create_from_options(options)

    def _create_polygon_mask(
        self, landmarks_px: np.ndarray, indices: list[int], shape: tuple[int, int]
    ) -> np.ndarray:
        """Create a filled polygon mask from landmark indices."""
        # Filter out indices that are beyond our landmark count
        valid_indices = [i for i in indices if i < len(landmarks_px)]
        if len(valid_indices) < 3:
            return np.zeros(shape, dtype=np.float32)

        points = landmarks_px[valid_indices].astype(np.int32)
        mask = np.zeros(shape, dtype=np.float32)
        cv2.fillPoly(mask, [points], 1.0)
        return mask

    def _create_region_masks(
        self, landmarks_px: np.ndarray, shape: tuple[int, int]
    ) -> FaceRegionMasks:
        """Generate masks for each facial region."""
        h, w = shape

        face_outline = self._create_polygon_mask(
            landmarks_px, LANDMARK_REGIONS["face_outline"], shape
        )
        left_eye = self._create_polygon_mask(
            landmarks_px, LANDMARK_REGIONS["left_eye_wide"], shape
        )
        right_eye = self._create_polygon_mask(
            landmarks_px, LANDMARK_REGIONS["right_eye_wide"], shape
        )
        nose = self._create_polygon_mask(
            landmarks_px, LANDMARK_REGIONS["nose"], shape
        )
        mouth = self._create_polygon_mask(
            landmarks_px, LANDMARK_REGIONS["mouth_tight"], shape
        )

        # Forehead: upper part of face outline, above nose bridge
        forehead = face_outline.copy()
        nose_indices = [i for i in LANDMARK_REGIONS["nose"] if i < len(landmarks_px)]
        if nose_indices:
            mid_y = int(landmarks_px[nose_indices[0], 1])
            forehead[mid_y:, :] = 0
        forehead = np.maximum(forehead - left_eye - right_eye, 0)

        # Cheeks: face area between eyes and mouth, left/right of nose
        nose_x_indices = [i for i in LANDMARK_REGIONS["nose"] if i < len(landmarks_px)]
        if nose_x_indices:
            center_x = int(np.mean(landmarks_px[nose_x_indices, 0]))
        else:
            center_x = w // 2

        cheek_base = face_outline.copy()
        cheek_base = np.maximum(
            cheek_base - left_eye - right_eye - nose - mouth - forehead, 0
        )
        left_cheek = cheek_base.copy()
        left_cheek[:, :center_x] = 0
        right_cheek = cheek_base.copy()
        right_cheek[:, center_x:] = 0

        # Chin: lower part of face below mouth
        mouth_indices = [i for i in LANDMARK_REGIONS["mouth_tight"] if i < len(landmarks_px)]
        if mouth_indices:
            chin_y = int(np.max(landmarks_px[mouth_indices, 1]))
        else:
            chin_y = h * 3 // 4
        chin = face_outline.copy()
        chin[:chin_y, :] = 0
        chin = np.maximum(chin - mouth, 0)

        combined = np.clip(
            face_outline + left_eye + right_eye + nose + mouth, 0, 1
        )

        return FaceRegionMasks(
            face_outline=face_outline,
            left_eye=left_eye,
            right_eye=right_eye,
            nose=nose,
            mouth=mouth,
            forehead=forehead,
            left_cheek=left_cheek,
            right_cheek=right_cheek,
            chin=chin,
            combined=combined,
        )

    def detect(self, image: Image.Image) -> list[FaceData]:
        """
        Detect faces and extract region masks.

        Args:
            image: PIL Image (RGB)

        Returns:
            List of FaceData for each detected face.
        """
        self._init_mediapipe()

        img_np = np.array(image)
        h, w = img_np.shape[:2]

        # Convert to MediaPipe Image
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_np)

        result = self._landmarker.detect(mp_image)

        if not result.face_landmarks:
            return []

        faces = []
        for face_landmarks in result.face_landmarks:
            # Extract normalized landmarks (x, y)
            landmarks = np.array(
                [(lm.x, lm.y) for lm in face_landmarks],
                dtype=np.float32,
            )

            # Convert to pixel coordinates
            pixel_landmarks = landmarks.copy()
            pixel_landmarks[:, 0] *= w
            pixel_landmarks[:, 1] *= h

            # Bounding box
            x_min = int(pixel_landmarks[:, 0].min())
            y_min = int(pixel_landmarks[:, 1].min())
            x_max = int(pixel_landmarks[:, 0].max())
            y_max = int(pixel_landmarks[:, 1].max())
            bbox = (x_min, y_min, x_max - x_min, y_max - y_min)

            # Create region masks
            region_masks = self._create_region_masks(pixel_landmarks, (h, w))

            # Confidence from detection
            confidence = 0.9

            faces.append(
                FaceData(
                    landmarks=landmarks,
                    pixel_landmarks=pixel_landmarks,
                    region_masks=region_masks,
                    bounding_box=bbox,
                    confidence=confidence,
                )
            )

        return faces
