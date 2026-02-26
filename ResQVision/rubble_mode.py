"""
ResQVision — Rubble Mode
=========================
Rubble region detection via texture analysis and
cross-referencing with YOLO person detections to flag trapped survivors.
"""

import cv2
import numpy as np


class RubbleDetector:
    """Identify rubble zones and detect potentially trapped survivors."""

    def __init__(
        self,
        edge_density_threshold: float = 0.12,
        contour_area_min: int = 3000,
    ):
        """
        Parameters
        ----------
        edge_density_threshold : float
            Minimum ratio of edge pixels to qualify as a rubble zone.
        contour_area_min : int
            Minimum contour area (px²) to be considered rubble.
        """
        self.edge_density_threshold = edge_density_threshold
        self.contour_area_min = contour_area_min

    def detect(self, frame: np.ndarray, person_bboxes: list) -> dict:
        """
        Detect rubble regions and identify trapped survivors.

        Parameters
        ----------
        frame : np.ndarray
            BGR frame.
        person_bboxes : list
            List of (x1, y1, x2, y2, conf) person detections.

        Returns
        -------
        dict
            {
                "rubble_zones": list[tuple],     # bounding rects of rubble
                "trapped_survivors": list[tuple], # person bboxes inside rubble
                "rubble_coverage": float,         # 0.0–1.0
                "rubble_mask": np.ndarray,
            }
        """
        h, w = frame.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # --- Texture analysis: Canny edges -----------------------------------
        edges = cv2.Canny(gray, 50, 150)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
        dilated = cv2.dilate(edges, kernel, iterations=2)

        # --- Find rubble contours --------------------------------------------
        contours, _ = cv2.findContours(
            dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        rubble_mask = np.zeros((h, w), dtype=np.uint8)
        rubble_zones = []

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < self.contour_area_min:
                continue
            x, y, cw, ch = cv2.boundingRect(cnt)
            # Edge density inside bounding rect
            roi_edges = edges[y : y + ch, x : x + cw]
            density = np.count_nonzero(roi_edges) / (cw * ch) if cw * ch > 0 else 0
            if density >= self.edge_density_threshold:
                rubble_zones.append((x, y, x + cw, y + ch))
                cv2.drawContours(rubble_mask, [cnt], -1, 255, -1)

        rubble_coverage = np.count_nonzero(rubble_mask) / (h * w)

        # --- Cross-reference persons with rubble zones -----------------------
        trapped_survivors = []
        for (px1, py1, px2, py2, conf) in person_bboxes:
            pcx, pcy = (px1 + px2) // 2, (py1 + py2) // 2
            for (rx1, ry1, rx2, ry2) in rubble_zones:
                if rx1 <= pcx <= rx2 and ry1 <= pcy <= ry2:
                    trapped_survivors.append((px1, py1, px2, py2, conf))
                    break

        return {
            "rubble_zones": rubble_zones,
            "trapped_survivors": trapped_survivors,
            "rubble_coverage": round(rubble_coverage, 4),
            "rubble_mask": rubble_mask,
        }

    def draw_overlay(self, frame: np.ndarray, result: dict) -> np.ndarray:
        """Draw rubble zones and trapped-survivor markers."""
        annotated = frame.copy()

        # Rubble zone tint
        overlay = annotated.copy()
        overlay[result["rubble_mask"] > 0] = (60, 60, 180)
        annotated = cv2.addWeighted(annotated, 0.75, overlay, 0.25, 0)

        # Rubble zone rectangles
        for (x1, y1, x2, y2) in result["rubble_zones"]:
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 140, 255), 2)
            cv2.putText(annotated, "RUBBLE", (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 140, 255), 1)

        # Trapped survivors
        for (x1, y1, x2, y2, conf) in result["trapped_survivors"]:
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 0, 255), 3)
            cv2.putText(annotated, "TRAPPED", (x1, y1 - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        return annotated
