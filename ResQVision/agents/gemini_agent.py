"""
ResQVision — Gemini Vision-Language Agent
===========================================
Multimodal AI for advanced disaster scene analysis.
"""

import os
import json
import base64
import typing
import numpy as np
import cv2
try:
    import google.generativeai as genai
except ImportError:
    genai = None

from PIL import Image

class GeminiAgent:
    """Uses Gemini 2.5 Flash to augment mathematical detection with semantic understanding."""

    def __init__(self, api_key: str | None = None):
        """
        Parameters
        ----------
        api_key : str, optional
            Google Gemini API key. If None, expects GEMINI_API_KEY env var.
        """
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY")
        self.configured = False
        self.model = None
        
        if self.api_key and genai is not None:
            self._configure()

    def _configure(self):
        try:
            genai.configure(api_key=self.api_key)
            
            # Using the 2.5 flash model as it supports multimodal (images + text)
            # and is extremely fast for real-time dashboard updates.
            self.model = genai.GenerativeModel(
                model_name="gemini-2.5-flash",
                system_instruction=(
                    "You are ResQ-AI, an expert Disaster Command Center Artificial Intelligence. "
                    "You analyze live camera feeds and telemetry data from drones/CCTVs in disaster zones "
                    "(earthquakes, floods, collapses, etc.). "
                    "Your job is to look at the provided image and telemetry, assess the TRUE danger level, "
                    "and provide an overriding Risk Score and a professional Situation Report (SITREP)."
                ),
                generation_config=genai.types.GenerationConfig(
                    temperature=0.2,
                    response_mime_type="application/json",
                )
            )
            self.configured = True
        except Exception as e:
            print(f"Failed to configure Gemini API: {e}")
            self.configured = False

    def update_key(self, api_key: str):
        """Update the API key at runtime (e.g., from Streamlit UI)."""
        self.api_key = api_key
        self._configure()

    def analyze(self, frame_bgr: np.ndarray, telemetry: dict) -> dict:
        """
        Analyze the scene using the image and mathematical telemetry.

        Parameters
        ----------
        frame_bgr : np.ndarray
            Raw OpenCV BGR frame.
        telemetry : dict
            Data from the standard pipeline (survivor_count, flood_risk, etc.)

        Returns
        -------
        dict
            A structured JSON response containing risk overrides and a SITREP.
        """
        if not self.configured or self.model is None:
            return self._fallback_response("Gemini API not configured. Please supply an API key.")

        try:
            # Convert BGR (OpenCV) to RGB (Pillow)
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(frame_rgb)

            # Construct the prompt with current dashboard math
            prompt = (
                f"Analyze this live disaster feed. The local rigid computer-vision pipeline reported:\n"
                f"- YOLO Survivor Count: {telemetry.get('survivor_count', 0)}\n"
                f"- Water Coverage %: {telemetry.get('water_coverage', 0.0) * 100:.1f}%\n"
                f"- Edge/Crowd Density %: {telemetry.get('crowd_density', 0.0) * 100:.1f}%\n"
                f"- Micro-Motion (trapped life): {telemetry.get('micro_motion_confidence', 0.0) * 100:.1f}%\n\n"
                f"Does the image confirm this data? What is ACTUALLY happening in the picture? "
                f"Are there hazards the computer missed (e.g. fire, blocked roads, massive destruction, tiny people in rubble)?\n\n"
                f"Return ONLY a JSON payload with exactly these keys:\n"
                f"1. \"overriding_risk_score\" (integer 0-100): Your intelligent assessment of the danger level.\n"
                f"2. \"sitrep_summary\" (string): A short, professional military/emergency style summary of the visual evidence (max 3 sentences).\n"
                f"3. \"tactical_advice\" (string): One clear recommendation for rescue teams based on the visual layout you see.\n"
            )

            response = self.model.generate_content([pil_img, prompt])
            
            # The model is configured to return JSON, so we can parse it
            result = json.loads(response.text)
            
            # Ensure required keys exist
            return {
                "overriding_risk_score": int(result.get("overriding_risk_score", telemetry.get("risk_score", 0))),
                "sitrep_summary": result.get("sitrep_summary", "No summary provided by AI."),
                "tactical_advice": result.get("tactical_advice", "No tactical advice generated."),
                "status": "success"
            }

        except Exception as e:
            print(f"Gemini API Error: {e}")
            return self._fallback_response(f"Gemini Analysis Failed: {str(e)}")

    def _fallback_response(self, error_msg: str) -> dict:
        return {
            "overriding_risk_score": None,
            "sitrep_summary": error_msg,
            "tactical_advice": "Awaiting valid API configuration/connection.",
            "status": "error"
        }
