"""
ResQVision — Micro-Motion Detection
=====================================
Detects subtle movement using frame differencing and
Farneback dense optical flow to identify signs of life.
"""

import cv2
import numpy as np


class MicroMotionDetector:
    """Detect micro-motion (subtle survivor movement) between consecutive frames."""

    def __init__(
        self,
        diff_threshold: int = 12,
        flow_magnitude_threshold: float = 0.4,
        min_motion_pixels: int = 30,
    ):
        """
        Parameters
        ----------
        diff_threshold : int
            Pixel intensity change threshold for frame differencing.
        flow_magnitude_threshold : float
            Minimum optical-flow magnitude to count as motion.
        min_motion_pixels : int
            Minimum moving pixels to register any motion at all.
        """
        self.diff_threshold = diff_threshold
        self.flow_mag_thresh = flow_magnitude_threshold
        self.min_motion_pixels = min_motion_pixels
        self._prev_gray: np.ndarray | None = None

    def detect(
        self,
        frame: np.ndarray,
        person_bboxes: list | None = None,
    ) -> dict:
        """
        Analyse micro-motion between the current and previous frame.

        Parameters
        ----------
        frame : np.ndarray
            Current BGR frame.
        person_bboxes : list, optional
            Person bounding boxes to restrict ROI analysis.

        Returns
        -------
        dict
            {
                "micro_motion_confidence": float,   # 0.0–1.0
                "diff_score": float,
                "flow_score": float,
                "motion_mask": np.ndarray | None,
            }
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)

        if self._prev_gray is None:
            self._prev_gray = gray
            return {
                "micro_motion_confidence": 0.0,
                "diff_score": 0.0,
                "flow_score": 0.0,
                "motion_mask": None,
            }

        # --- Frame differencing ----------------------------------------------
        diff = cv2.absdiff(self._prev_gray, gray)
        _, motion_mask = cv2.threshold(
            diff, self.diff_threshold, 255, cv2.THRESH_BINARY
        )

        diff_pixels = np.count_nonzero(motion_mask)
        total_pixels = gray.shape[0] * gray.shape[1]
        diff_score = min(diff_pixels / max(total_pixels * 0.01, 1), 1.0)

        # --- Farneback optical flow ------------------------------------------
        flow = cv2.calcOpticalFlowFarneback(
            self._prev_gray, gray,
            None, 0.5, 3, 15, 3, 5, 1.2, 0
        )
        mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])

        # If person bboxes given, focus on those ROIs
        if person_bboxes:
            roi_mags = []
            for (x1, y1, x2, y2, *_) in person_bboxes:
                x1, y1 = max(0, x1), max(0, y1)
                x2 = min(mag.shape[1], x2)
                y2 = min(mag.shape[0], y2)
                roi = mag[y1:y2, x1:x2]
                if roi.size > 0:
                    roi_mags.append(np.mean(roi))
            avg_mag = float(np.mean(roi_mags)) if roi_mags else 0.0
        else:
            avg_mag = float(np.mean(mag))

        flow_score = min(avg_mag / 2.0, 1.0)  # normalise

        # --- Combined confidence ---------------------------------------------
        if diff_pixels < self.min_motion_pixels:
            confidence = 0.0
        else:
            confidence = 0.5 * diff_score + 0.5 * flow_score

        self._prev_gray = gray

        return {
            "micro_motion_confidence": round(confidence, 4),
            "diff_score": round(diff_score, 4),
            "flow_score": round(flow_score, 4),
            "motion_mask": motion_mask,
        }
