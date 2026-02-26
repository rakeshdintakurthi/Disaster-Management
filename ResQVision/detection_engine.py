"""
ResQVision — Detection Engine
==============================
Core YOLOv8-based person detection pipeline with:
 • Multi-frame confirmation (sliding window)
 • Exponential moving average confidence smoothing
 • False-positive suppression
 • Probability heatmap overlay
 • Snapshot + CSV logging
"""

import os
import csv
import time
import numpy as np
import cv2
from datetime import datetime
from collections import deque

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False


class DetectionEngine:
    """YOLOv8  person detector with multi-frame confirmation."""

    # COCO class index for 'person'
    PERSON_CLASS = 0

    def __init__(
        self,
        model_path: str = "yolov8m.pt",
        confidence_threshold: float = 0.25,
        confirmation_window: int = 20,
        confirmation_min: int = 15,
        ema_alpha: float = 0.3,
        log_dir: str = "logs",
    ):
        # --- Model -----------------------------------------------------------
        if YOLO_AVAILABLE:
            self.model = YOLO(model_path)
        else:
            self.model = None
            print("[WARN] ultralytics not installed — running in DEMO mode.")

        self.confidence_threshold = confidence_threshold

        # --- Multi-frame confirmation ----------------------------------------
        self.confirmation_window = confirmation_window
        self.confirmation_min = confirmation_min
        # sliding window: each entry is (count, avg_confidence)
        self._history: deque = deque(maxlen=confirmation_window)

        # --- Confidence smoothing --------------------------------------------
        self.ema_alpha = ema_alpha
        self._smoothed_confidence: float = 0.0

        # --- Heatmap ---------------------------------------------------------
        self._heatmap: np.ndarray | None = None

        # --- Logging ---------------------------------------------------------
        self.log_dir = log_dir
        self.snapshot_dir = os.path.join(log_dir, "snapshots")
        os.makedirs(self.snapshot_dir, exist_ok=True)
        self.csv_path = os.path.join(log_dir, "detections.csv")
        self._init_csv()

        # --- Counters --------------------------------------------------------
        self._frame_idx: int = 0
        self._confirmed_count: int = 0

    # ------------------------------------------------------------------
    # CSV helpers
    # ------------------------------------------------------------------

    def _init_csv(self):
        """Create CSV with header if it doesn't exist."""
        if not os.path.exists(self.csv_path):
            with open(self.csv_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "frame", "timestamp", "raw_count", "confirmed_count",
                    "avg_confidence", "smoothed_confidence",
                ])

    def _log_csv(self, raw_count: int, avg_conf: float):
        with open(self.csv_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                self._frame_idx,
                datetime.now().isoformat(),
                raw_count,
                self._confirmed_count,
                round(avg_conf, 4),
                round(self._smoothed_confidence, 4),
            ])

    # ------------------------------------------------------------------
    # Heatmap
    # ------------------------------------------------------------------

    def _update_heatmap(self, frame_shape: tuple, centroids: list):
        """Accumulate Gaussian blobs at detection centroids."""
        h, w = frame_shape[:2]
        if self._heatmap is None or self._heatmap.shape[:2] != (h, w):
            self._heatmap = np.zeros((h, w), dtype=np.float32)

        for (cx, cy) in centroids:
            # Draw a Gaussian blob (radius ~40 px)
            cv2.circle(self._heatmap, (int(cx), int(cy)), 40, 1.0, -1)

        # Slight decay so old detections fade
        self._heatmap *= 0.97

    def get_heatmap_overlay(self, frame: np.ndarray) -> np.ndarray:
        """Return frame blended with the probability heatmap."""
        if self._heatmap is None:
            return frame
        norm = cv2.normalize(self._heatmap, None, 0, 255, cv2.NORM_MINMAX)
        colored = cv2.applyColorMap(norm.astype(np.uint8), cv2.COLORMAP_JET)
        return cv2.addWeighted(frame, 0.7, colored, 0.3, 0)

    # ------------------------------------------------------------------
    # Core detection
    # ------------------------------------------------------------------

    def detect(self, frame: np.ndarray, is_photo: bool = False) -> dict:
        """
        Run detection on a single frame.

        Returns
        -------
        dict
            {
                "survivor_count": int,        # confirmed count
                "raw_count": int,
                "confidence_score": float,    # smoothed
                "bboxes": list[tuple],        # (x1, y1, x2, y2, conf)
                "centroids": list[tuple],
                "confirmed": bool,
                "frame_index": int,
            }
        """
        self._frame_idx += 1
        bboxes = []
        centroids = []
        confidences = []

        # --- Run YOLO --------------------------------------------------------
        if self.model is not None:
            # Enforce larger imgsz for better detection of tiny distant people
            inference_size = 2560 if is_photo else 1280
            results = self.model(frame, imgsz=inference_size, verbose=False)
            for r in results:
                for box in r.boxes:
                    cls_id = int(box.cls[0])
                    conf = float(box.conf[0])
                    if cls_id == self.PERSON_CLASS and conf >= self.confidence_threshold:
                        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                        bboxes.append((x1, y1, x2, y2, conf))
                        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                        centroids.append((cx, cy))
                        confidences.append(conf)
        else:
            # DEMO fallback — generate synthetic detections for testing
            h, w = frame.shape[:2]
            demo_count = np.random.randint(0, 4)
            for _ in range(demo_count):
                x1 = np.random.randint(0, w - 80)
                y1 = np.random.randint(0, h - 120)
                x2, y2 = x1 + 60, y1 + 100
                conf = round(np.random.uniform(0.45, 0.95), 2)
                bboxes.append((x1, y1, x2, y2, conf))
                centroids.append(((x1 + x2) // 2, (y1 + y2) // 2))
                confidences.append(conf)

        raw_count = len(bboxes)
        avg_conf = float(np.mean(confidences)) if confidences else 0.0

        # --- Multi-frame confirmation ----------------------------------------
        self._history.append((raw_count, avg_conf))
        
        if is_photo:
            # Bypass confirmation logic if we are just analysing a static photo
            confirmed = True
            self._confirmed_count = raw_count
        else:
            frames_with_detections = sum(1 for (c, _) in self._history if c > 0)
            confirmed = frames_with_detections >= self.confirmation_min

            if confirmed:
                # Use median count from window for stability
                counts = [c for (c, _) in self._history if c > 0]
                self._confirmed_count = int(np.median(counts))
            else:
                self._confirmed_count = 0

        # --- EMA confidence smoothing ----------------------------------------
        self._smoothed_confidence = (
            self.ema_alpha * avg_conf
            + (1 - self.ema_alpha) * self._smoothed_confidence
        )

        # --- Heatmap update --------------------------------------------------
        self._update_heatmap(frame.shape, centroids)

        # --- Snapshot on new confirmed detections ----------------------------
        if confirmed and self._confirmed_count > 0 and self._frame_idx % 30 == 0:
            snap_path = os.path.join(
                self.snapshot_dir,
                f"snap_{self._frame_idx}_{datetime.now().strftime('%H%M%S')}.jpg",
            )
            cv2.imwrite(snap_path, frame)

        # --- CSV logging -----------------------------------------------------
        self._log_csv(raw_count, avg_conf)

        return {
            "survivor_count": self._confirmed_count,
            "raw_count": raw_count,
            "confidence_score": round(self._smoothed_confidence, 4),
            "bboxes": bboxes,
            "centroids": centroids,
            "confirmed": confirmed,
            "frame_index": self._frame_idx,
        }

    # ------------------------------------------------------------------
    # Drawing helper
    # ------------------------------------------------------------------

    def draw_detections(self, frame: np.ndarray, result: dict) -> np.ndarray:
        """Draw bounding boxes and info text on frame."""
        annotated = frame.copy()
        color = (0, 255, 0) if result["confirmed"] else (0, 165, 255)

        for (x1, y1, x2, y2, conf) in result["bboxes"]:
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            label = f"Person {conf:.0%}"
            cv2.putText(annotated, label, (x1, y1 - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        # Status bar
        status = (
            f"Survivors: {result['survivor_count']} | "
            f"Conf: {result['confidence_score']:.2%} | "
            f"Frame: {result['frame_index']}"
        )
        cv2.putText(annotated, status, (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        return annotated
