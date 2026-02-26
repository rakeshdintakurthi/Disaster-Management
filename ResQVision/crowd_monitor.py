"""
ResQVision — Crowd Monitor
============================
Crowd density estimation based on detection count and occupied area.
"""

import cv2
import numpy as np


class CrowdMonitor:
    """Estimate crowd density from YOLO person detections."""

    def __init__(
        self,
        frame_area: int = 640 * 480,
        max_expected_persons: int = 20,
    ):
        """
        Parameters
        ----------
        frame_area : int
            Total pixel area of the frame (used for area-based density).
        max_expected_persons : int
            Upper cap for normalising count-based density.
        """
        self.frame_area = frame_area
        self.max_expected_persons = max_expected_persons

    def estimate(self, bboxes: list, frame_shape: tuple | None = None, frame: np.ndarray | None = None) -> dict:
        """
        Estimate crowd density.

        Parameters
        ----------
        bboxes : list
            List of (x1, y1, x2, y2, conf) detections.
        frame_shape : tuple, optional
            (H, W, C) — if given, recalculates frame_area.
        frame : np.ndarray, optional
            Raw BGR frame used for edge-based fallback.

        Returns
        -------
        dict
            {
                "crowd_density": float,   # 0.0–1.0 normalised
                "person_count": int,
                "occupied_area_ratio": float,
            }
        """
        # --- Fallback: Edge density for aerial drone shots -------------------
        # In dense aerial shots, YOLO might fail completely (0 bboxes).
        # We can detect 'chaos' (dense crowds) using Canny edges.
        edge_density = 0.0
        if frame is not None:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # Blur slightly to remove tiny noise
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            edges = cv2.Canny(blurred, 50, 150)
            
            # What percentage of pixels are edges?
            h, w = edges.shape
            edge_pixels = np.count_nonzero(edges)
            # Typically, >10% edge pixels in a scene means EXTREME clutter/crowd 
            # (especially in empty street / flood scenarios)
            edge_density = min(edge_pixels / (h * w * 0.10), 1.0)
            
        if frame_shape is not None:
            self.frame_area = frame_shape[0] * frame_shape[1]

        count = len(bboxes)

        # Area occupied by all bounding boxes (rough, ignores overlap)
        occupied = 0
        for (x1, y1, x2, y2, *_) in bboxes:
            occupied += abs(x2 - x1) * abs(y2 - y1)

        area_ratio = min(occupied / self.frame_area, 1.0) if self.frame_area > 0 else 0.0
        count_ratio = min(count / self.max_expected_persons, 1.0)

        # Combined density: 60 % count-based + 40 % area-based
        density = 0.6 * count_ratio + 0.4 * area_ratio

        # If YOLO fails but the scene is highly chaotic (high edge density), 
        # use the edge density as a fallback to trigger risk logic.
        if density < 0.1 and edge_density > 0.4:
            density = edge_density * 0.8  # Apply 80% weight to edge fallback

        return {
            "crowd_density": round(density, 4),
            "person_count": count,
            "occupied_area_ratio": round(area_ratio, 4),
        }
