"""
ResQVision — Frame Preprocessing Module
========================================
Provides frame-level utilities: resize, normalise, denoise,
CLAHE contrast enhancement, and edge detection helpers.
"""

import cv2
import numpy as np


class FramePreprocessor:
    """Preprocessing pipeline applied to every incoming video frame."""

    def __init__(self, target_width: int = 640, target_height: int = 480):
        self.target_width = target_width
        self.target_height = target_height
        # CLAHE for adaptive contrast enhancement
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

    # ------------------------------------------------------------------
    # Core transforms
    # ------------------------------------------------------------------

    def resize(self, frame: np.ndarray) -> np.ndarray:
        """Resize frame to target dimensions."""
        return cv2.resize(frame, (self.target_width, self.target_height),
                          interpolation=cv2.INTER_LINEAR)

    def to_gray(self, frame: np.ndarray) -> np.ndarray:
        """Convert BGR frame to grayscale."""
        if len(frame.shape) == 3 and frame.shape[2] == 3:
            return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return frame

    def denoise(self, frame: np.ndarray, method: str = "gaussian") -> np.ndarray:
        """
        Denoise a frame.
        Methods: 'gaussian' (fast) | 'bilateral' (edge-preserving).
        """
        if method == "bilateral":
            return cv2.bilateralFilter(frame, d=9, sigmaColor=75, sigmaSpace=75)
        return cv2.GaussianBlur(frame, (5, 5), 0)

    def enhance_contrast(self, frame: np.ndarray) -> np.ndarray:
        """Apply CLAHE contrast enhancement (works on grayscale)."""
        gray = self.to_gray(frame)
        return self.clahe.apply(gray)

    def normalize(self, frame: np.ndarray) -> np.ndarray:
        """Min-max normalise pixel values to [0, 255]."""
        return cv2.normalize(frame, None, 0, 255, cv2.NORM_MINMAX)

    def detect_edges(self, frame: np.ndarray,
                     low: int = 50, high: int = 150) -> np.ndarray:
        """Canny edge detection helper."""
        gray = self.to_gray(frame)
        return cv2.Canny(gray, low, high)

    # ------------------------------------------------------------------
    # Full pipeline
    # ------------------------------------------------------------------

    def preprocess(self, frame: np.ndarray,
                   denoise_method: str = "gaussian") -> np.ndarray:
        """
        Run the standard preprocessing pipeline:
        resize → denoise → return BGR frame ready for detection.
        """
        frame = self.resize(frame)
        frame = self.denoise(frame, method=denoise_method)
        return frame
