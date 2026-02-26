"""
ResQVision — Breathing Analysis
================================
Detects periodic chest-region pixel-intensity changes
to infer potential breathing patterns (FFT-based).
"""

import cv2
import numpy as np
from collections import deque


class BreathingAnalyzer:
    """Analyse subtle periodic intensity changes indicative of breathing."""

    # Typical breathing: 9–30 breaths/min → 0.15–0.50 Hz
    BREATHING_LOW_HZ = 0.15
    BREATHING_HIGH_HZ = 0.50

    def __init__(self, fps: float = 30.0, buffer_seconds: float = 5.0):
        """
        Parameters
        ----------
        fps : float
            Video frame rate (needed for frequency conversion).
        buffer_seconds : float
            How many seconds of intensity history to keep for FFT.
        """
        self.fps = fps
        self.buffer_size = int(fps * buffer_seconds)
        # Per-person intensity buffers keyed by person_id (index)
        self._buffers: dict[int, deque] = {}

    def _chest_roi(self, bbox: tuple) -> tuple:
        """Estimate a chest region from a person bounding box."""
        x1, y1, x2, y2 = bbox[:4]
        h = y2 - y1
        # Chest ≈ 30–60 % of bbox height
        cy1 = y1 + int(h * 0.30)
        cy2 = y1 + int(h * 0.60)
        return (x1, cy1, x2, cy2)

    def analyse(
        self,
        frame: np.ndarray,
        person_bboxes: list,
    ) -> dict:
        """
        Analyse breathing for each detected person.

        Returns
        -------
        dict
            {
                "breathing_confidence": float,          # max across persons
                "per_person": list[dict],               # per-person results
            }
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        per_person = []

        for idx, bbox in enumerate(person_bboxes):
            x1, cy1, x2, cy2 = self._chest_roi(bbox)

            # Clamp to frame
            x1, cy1 = max(0, x1), max(0, cy1)
            x2 = min(gray.shape[1], x2)
            cy2 = min(gray.shape[0], cy2)

            roi = gray[cy1:cy2, x1:x2]
            if roi.size == 0:
                per_person.append({"person_id": idx, "breathing_conf": 0.0})
                continue

            mean_intensity = float(np.mean(roi))

            # Maintain per-person buffer
            if idx not in self._buffers:
                self._buffers[idx] = deque(maxlen=self.buffer_size)
            self._buffers[idx].append(mean_intensity)

            buf = self._buffers[idx]
            if len(buf) < self.buffer_size // 2:
                # Not enough data yet
                per_person.append({"person_id": idx, "breathing_conf": 0.0})
                continue

            # --- FFT analysis ------------------------------------------------
            signal = np.array(buf) - np.mean(buf)  # remove DC
            fft_vals = np.abs(np.fft.rfft(signal))
            freqs = np.fft.rfftfreq(len(signal), d=1.0 / self.fps)

            # Mask to breathing band
            mask = (freqs >= self.BREATHING_LOW_HZ) & (freqs <= self.BREATHING_HIGH_HZ)
            breathing_energy = np.sum(fft_vals[mask])
            total_energy = np.sum(fft_vals[1:]) + 1e-9  # skip DC

            breathing_conf = min(breathing_energy / total_energy, 1.0)
            per_person.append({
                "person_id": idx,
                "breathing_conf": round(float(breathing_conf), 4),
            })

        max_conf = max((p["breathing_conf"] for p in per_person), default=0.0)

        return {
            "breathing_confidence": round(max_conf, 4),
            "per_person": per_person,
        }
