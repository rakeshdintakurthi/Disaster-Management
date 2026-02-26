"""
ResQVision — Main Pipeline Orchestrator
=========================================
Ties together all detection, analysis, and agent modules
into a single processing pipeline.

Usage
-----
Console mode:
    python main.py                    # webcam
    python main.py --source video.mp4 # video file

The function `run_pipeline_frame()` is consumed by the
Streamlit dashboard for per-frame processing.
"""

import os
import sys
import argparse
import time
import cv2
import numpy as np

# ── Local module imports ────────────────────────────────────────────────
from preprocessing import FramePreprocessor
from detection_engine import DetectionEngine
from flood_mode import FloodDetector
from rubble_mode import RubbleDetector
from crowd_monitor import CrowdMonitor
from micro_motion import MicroMotionDetector
from breathing_analysis import BreathingAnalyzer

from agents.risk_agent import RiskAgent
from agents.strategy_agent import StrategyAgent
from agents.resource_agent import ResourceAgent
from agents.report_agent import ReportAgent
from agents.gemini_agent import GeminiAgent


# ═══════════════════════════════════════════════════════════════════════
# Pipeline class
# ═══════════════════════════════════════════════════════════════════════

class ResQVisionPipeline:
    """End-to-end disaster intelligence pipeline."""

    def __init__(self, mode: str = "flood", log_dir: str = "logs"):
        """
        Parameters
        ----------
        mode : str
            'flood' or 'rubble' — selects the scenario module.
        log_dir : str
            Directory for logs, snapshots, and reports.
        """
        os.makedirs(log_dir, exist_ok=True)
        self.mode = mode
        self.log_dir = log_dir

        # ── Modules ──────────────────────────────────────────────────
        self.preprocessor = FramePreprocessor()
        self.detector = DetectionEngine(log_dir=log_dir)
        self.flood_detector = FloodDetector()
        self.rubble_detector = RubbleDetector()
        self.crowd_monitor = CrowdMonitor()
        self.motion_detector = MicroMotionDetector()
        self.breathing_analyzer = BreathingAnalyzer()

        # ── Agents ───────────────────────────────────────────────────
        self.risk_agent = RiskAgent()
        self.strategy_agent = StrategyAgent()
        self.resource_agent = ResourceAgent()
        self.report_agent = ReportAgent(report_dir=self.log_dir)
        self.gemini_agent = GeminiAgent()

        # ── State ────────────────────────────────────────────────────
        self.last_result: dict = {}
        self.frame_count: int = 0
        self.fps: float = 0.0
        self._tick = time.time()

    # ──────────────────────────────────────────────────────────────────
    # Per-frame processing (used by both console & dashboard)
    # ──────────────────────────────────────────────────────────────────

    def run_pipeline_frame(self, raw_frame: np.ndarray, is_photo: bool = False) -> dict:
        """
        Process a single frame through the full pipeline.

        Returns a dict with all aggregated results.
        """
        self.frame_count += 1
        now = time.time()
        dt = now - self._tick
        self.fps = 1.0 / dt if dt > 0 else 0.0
        self._tick = now

        # 1 — Preprocess
        frame = self.preprocessor.preprocess(raw_frame)

        # 2 — Person detection
        det = self.detector.detect(frame, is_photo=is_photo)

        # 3 — Scenario mode
        flood_result = {"flood_risk": False, "water_coverage": 0.0,
                        "flood_line_y": 0, "water_mask": np.zeros(frame.shape[:2], dtype=np.uint8)}
        rubble_result = {"rubble_zones": [], "trapped_survivors": [],
                         "rubble_coverage": 0.0, "rubble_mask": np.zeros(frame.shape[:2], dtype=np.uint8)}

        if self.mode == "flood":
            flood_result = self.flood_detector.detect(frame)
        elif self.mode == "rubble":
            rubble_result = self.rubble_detector.detect(frame, det["bboxes"])

        # 4 — Crowd density
        crowd = self.crowd_monitor.estimate(det["bboxes"], frame_shape=frame.shape, frame=frame)

        # 5 — Micro-motion
        motion = self.motion_detector.detect(frame, det["bboxes"])

        # 6 — Breathing analysis
        breathing = self.breathing_analyzer.analyse(frame, det["bboxes"])

        # ── Aggregate data dict for agents ──────────────────────────
        agent_input = {
            "survivor_count": det["survivor_count"],
            "confidence_score": det["confidence_score"],
            "flood_risk": flood_result["flood_risk"],
            "water_coverage": flood_result["water_coverage"],
            "crowd_density": crowd["crowd_density"],
            "micro_motion_confidence": motion["micro_motion_confidence"],
        }

        # 7 — Risk Agent
        risk = self.risk_agent.assess(agent_input)

        # 8 — Strategy Agent
        strategy = self.strategy_agent.recommend(risk["risk_level"])

        # 9 — Resource Agent
        resources = self.resource_agent.allocate(
            risk["risk_level"], det["survivor_count"], crowd["crowd_density"]
        )

        # 10 — Report Agent (generate every 100 frames or when risk changes)
        report_text = ""
        if self.frame_count % 100 == 0 or self.frame_count == 1:
            report_text = self.report_agent.generate(
                survivor_count=det["survivor_count"],
                risk_level=risk["risk_level"],
                risk_score=risk["risk_score"],
                strategy=strategy,
                resources=resources,
                confidence_score=det["confidence_score"],
                flood_risk=flood_result["flood_risk"],
                micro_motion_confidence=motion["micro_motion_confidence"],
                breathing_confidence=breathing["breathing_confidence"],
            )

        # ── Build annotated frame ───────────────────────────────────
        annotated = self.detector.draw_detections(frame, det)
        if self.mode == "flood":
            annotated = self.flood_detector.draw_overlay(annotated, flood_result)
        elif self.mode == "rubble":
            annotated = self.rubble_detector.draw_overlay(annotated, rubble_result)

        # 11 — Gemini AI Analysis (Throttle to every 30 frames for video, or every frame for photo)
        gemini_result = {
            "overriding_risk_score": None,
            "sitrep_summary": "Awaiting API / analyzing...",
            "tactical_advice": "",
            "status": "pending",
        }
        if is_photo or (self.frame_count % 30 == 0):
            # Pass the raw annotated image and the math telemetry to the LLM
            gemini_result = self.gemini_agent.analyze(annotated, risk)


        heatmap_frame = self.detector.get_heatmap_overlay(frame)

        # ── Package result ──────────────────────────────────────────
        self.last_result = {
            "frame": frame,
            "raw_frame": raw_frame,
            "annotated_frame": annotated,
            "heatmap_frame": heatmap_frame,
            "detection": det,
            "flood": flood_result,
            "rubble": rubble_result,
            "crowd": crowd,
            "motion": motion,
            "breathing": breathing,
            "risk": risk,
            "strategy": strategy,
            "resources": resources,
            "report": report_text,
            "fps": round(self.fps, 1),
            "frame_count": self.frame_count,
            "mode": self.mode,
        }
        return self.last_result


# ═══════════════════════════════════════════════════════════════════════
# Console entry point
# ═══════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="ResQVision — Console Pipeline")
    parser.add_argument("--source", type=str, default="0",
                        help="Video source: 0 for webcam, or path to video file")
    parser.add_argument("--mode", type=str, default="flood",
                        choices=["flood", "rubble"],
                        help="Scenario mode")
    args = parser.parse_args()

    source = int(args.source) if args.source.isdigit() else args.source
    cap = cv2.VideoCapture(source)

    if not cap.isOpened():
        print(f"[ERROR] Cannot open video source: {args.source}")
        sys.exit(1)

    pipeline = ResQVisionPipeline(mode=args.mode)
    print(f"[ResQVision] Pipeline started — mode={args.mode}, source={args.source}")
    print("[ResQVision] Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[ResQVision] End of video / no frame captured.")
            break

        result = pipeline.run_pipeline_frame(frame)

        # Show annotated feed
        cv2.imshow("ResQVision — Detection Feed", result["annotated_frame"])
        cv2.imshow("ResQVision — Heatmap", result["heatmap_frame"])

        # Console summary every 30 frames
        if result["frame_count"] % 30 == 0:
            r = result
            print(
                f"[Frame {r['frame_count']}] "
                f"Survivors: {r['detection']['survivor_count']} | "
                f"Conf: {r['detection']['confidence_score']:.2%} | "
                f"Risk: {r['risk']['risk_level']} ({r['risk']['risk_score']}) | "
                f"Motion: {r['motion']['micro_motion_confidence']:.2%} | "
                f"FPS: {r['fps']}"
            )

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("[ResQVision] Pipeline stopped.")


if __name__ == "__main__":
    main()
