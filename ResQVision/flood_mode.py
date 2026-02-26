"""
ResQVision — Flood Mode
========================
HSV-based water segmentation and flood-line detection.
"""

import cv2
import numpy as np


class FloodDetector:
    """Detect water regions and determine flood risk."""

    def __init__(
        self,
        flood_line_ratio: float = 0.5,
        water_coverage_threshold: float = 0.15,
    ):
        """
        Parameters
        ----------
        flood_line_ratio : float
            Y-position ratio (0=top, 1=bottom). If water is above this line → high risk.
        water_coverage_threshold : float
            Fraction of frame covered by water to flag flood_risk.
        """
        self.flood_line_ratio = flood_line_ratio
        self.water_coverage_threshold = water_coverage_threshold

        # HSV ranges for water
        # We need two ranges: one for blue/teal water, one for muddy/brown water
        self.lower_blue = np.array([85, 40, 40])
        self.upper_blue = np.array([135, 255, 255])
        
        self.lower_muddy = np.array([10, 20, 40])
        self.upper_muddy = np.array([35, 255, 200])

    def detect(self, frame: np.ndarray) -> dict:
        """
        Analyse a frame for flood indicators.

        Returns
        -------
        dict
            {
                "flood_risk": bool,
                "water_coverage": float,       # 0.0–1.0
                "flood_line_y": int,            # pixel Y of the threshold line
                "water_mask": np.ndarray,       # binary mask
            }
        """
        h, w = frame.shape[:2]
        flood_line_y = int(h * self.flood_line_ratio)

        # --- Water segmentation (HSV) ----------------------------------------
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask_blue = cv2.inRange(hsv, self.lower_blue, self.upper_blue)
        mask_muddy = cv2.inRange(hsv, self.lower_muddy, self.upper_muddy)
        mask = cv2.bitwise_or(mask_blue, mask_muddy)

        # Morphological clean-up
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

        water_pixels = np.count_nonzero(mask)
        total_pixels = h * w
        water_coverage = water_pixels / total_pixels

        # --- Flood risk assessment -------------------------------------------
        # Check if significant water exists above the flood line
        upper_water = np.count_nonzero(mask[:flood_line_y, :])
        upper_total = flood_line_y * w
        upper_coverage = upper_water / upper_total if upper_total > 0 else 0

        flood_risk = (
            water_coverage >= self.water_coverage_threshold
            or upper_coverage >= 0.10
        )

        return {
            "flood_risk": flood_risk,
            "water_coverage": round(water_coverage, 4),
            "flood_line_y": flood_line_y,
            "water_mask": mask,
        }

    def draw_overlay(self, frame: np.ndarray, result: dict) -> np.ndarray:
        """Draw flood line and water tint on frame."""
        annotated = frame.copy()
        h, w = annotated.shape[:2]
        flood_y = result["flood_line_y"]

        # Flood line
        color = (0, 0, 255) if result["flood_risk"] else (0, 200, 200)
        cv2.line(annotated, (0, flood_y), (w, flood_y), color, 2)
        cv2.putText(annotated, "FLOOD LINE", (10, flood_y - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        # Semi-transparent water tint
        water_overlay = annotated.copy()
        water_overlay[result["water_mask"] > 0] = (200, 120, 40)
        annotated = cv2.addWeighted(annotated, 0.7, water_overlay, 0.3, 0)

        # Risk badge
        if result["flood_risk"]:
            cv2.putText(annotated, "⚠ FLOOD RISK", (w - 200, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        return annotated
