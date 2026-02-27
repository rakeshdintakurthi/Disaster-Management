"""
ResQVision — Fire Mode
========================
HSV-based fire and smoke segmentation.
"""

import cv2
import numpy as np


class FireDetector:
    """Detect fire regions and determine fire risk."""

    def __init__(
        self,
        fire_coverage_threshold: float = 0.05,
    ):
        """
        Parameters
        ----------
        fire_coverage_threshold : float
            Fraction of frame covered by fire to flag fire_risk.
        """
        self.fire_coverage_threshold = fire_coverage_threshold

        # HSV ranges for fire (bright yellow/orange/red)
        # We need ranges that capture the core of the fire and the edges
        self.lower_fire1 = np.array([0, 100, 100])
        self.upper_fire1 = np.array([20, 255, 255])
        
        self.lower_fire2 = np.array([160, 100, 100])
        self.upper_fire2 = np.array([180, 255, 255])
        
        # Ranges for white-hot center of fire
        self.lower_white = np.array([0, 0, 200])
        self.upper_white = np.array([180, 50, 255])

    def detect(self, frame: np.ndarray) -> dict:
        """
        Analyse a frame for fire indicators.

        Returns
        -------
        dict
            {
                "fire_risk": bool,
                "fire_coverage": float,       # 0.0–1.0
                "fire_mask": np.ndarray,       # binary mask
            }
        """
        h, w = frame.shape[:2]

        # --- Fire segmentation (HSV) ----------------------------------------
        # Blur the image slightly to reduce noise
        blurred = cv2.GaussianBlur(frame, (11, 11), 0)
        hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
        
        # Create masks for the different hue ranges of fire
        mask1 = cv2.inRange(hsv, self.lower_fire1, self.upper_fire1)
        mask2 = cv2.inRange(hsv, self.lower_fire2, self.upper_fire2)
        mask3 = cv2.inRange(hsv, self.lower_white, self.upper_white)
        
        # Combine masks
        mask = cv2.bitwise_or(mask1, mask2)
        mask = cv2.bitwise_or(mask, mask3)

        # Morphological clean-up to remove small noise and fill holes
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        fire_pixels = np.count_nonzero(mask)
        total_pixels = h * w
        fire_coverage = fire_pixels / total_pixels

        # --- Fire risk assessment -------------------------------------------
        fire_risk = fire_coverage >= self.fire_coverage_threshold

        return {
            "fire_risk": fire_risk,
            "fire_coverage": round(fire_coverage, 4),
            "fire_mask": mask,
        }

    def draw_overlay(self, frame: np.ndarray, result: dict) -> np.ndarray:
        """Draw fire bounding boxes and tint on frame."""
        annotated = frame.copy()
        
        if result["fire_coverage"] > 0:
            # Find contours of the fire regions
            contours, _ = cv2.findContours(result["fire_mask"], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Draw bounding boxes around significant fire regions
            for cnt in contours:
                area = cv2.contourArea(cnt)
                if area > 500: # Filter out tiny noise
                    x, y, w, h = cv2.boundingRect(cnt)
                    cv2.rectangle(annotated, (x, y), (x+w, y+h), (0, 165, 255), 2) # Orange box
                    
                    # Add a semi-transparent orange tint over the actual fire mask
                    fire_overlay = annotated.copy()
                    fire_overlay[result["fire_mask"] > 0] = (0, 69, 255) # BGR Orange-Red
                    annotated = cv2.addWeighted(annotated, 0.6, fire_overlay, 0.4, 0)
        
        # Risk badge
        if result["fire_risk"]:
            h, w = annotated.shape[:2]
            cv2.putText(annotated, "⚠ FIRE DETECTED", (w - 230, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        return annotated
