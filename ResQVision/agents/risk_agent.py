"""
ResQVision — Risk Assessment Agent
====================================
Weighted multi-factor risk scoring.
"""


class RiskAgent:
    """Evaluate disaster risk level from aggregated detection data."""

    # Factor weights (must sum to 1.0)
    WEIGHTS = {
        "survivor_count": 0.15,
        "confidence_score": 0.05,
        "flood_risk": 0.15,
        "fire_risk": 0.20,
        "crowd_density": 0.30,
        "micro_motion_confidence": 0.15,
    }

    # Risk level thresholds
    THRESHOLDS = [
        (80, "CRITICAL"),
        (55, "HIGH"),
        (30, "MEDIUM"),
        (0, "LOW"),
    ]

    def assess(self, data: dict) -> dict:
        """
        Assess disaster risk.

        Parameters
        ----------
        data : dict
            Expected keys:
            - survivor_count (int)
            - confidence_score (float, 0–1)
            - flood_risk (bool)
            - water_coverage (float, 0-1)
            - fire_risk (bool)
            - fire_coverage (float, 0-1)
            - crowd_density (float, 0–1)
            - micro_motion_confidence (float, 0–1)

        Returns
        -------
        dict
            {
                "risk_level": str,   # LOW / MEDIUM / HIGH / CRITICAL
                "risk_score": int,   # 0–100
                "breakdown": dict,   # per-factor scores
            }
        """
        # Normalise survivor count (cap at 10 for max score)
        surv_norm = min(data.get("survivor_count", 0) / 10, 1.0)
        conf = data.get("confidence_score", 0.0)
        flood = 1.0 if data.get("flood_risk", False) else 0.0
        fire = 1.0 if data.get("fire_risk", False) else 0.0
        crowd = data.get("crowd_density", 0.0)
        motion = data.get("micro_motion_confidence", 0.0)

        breakdown = {
            "survivor_count": round(surv_norm * 100, 1),
            "confidence_score": round(conf * 100, 1),
            "flood_risk": round(flood * 100, 1),
            "fire_risk": round(fire * 100, 1),
            "crowd_density": round(crowd * 100, 1),
            "micro_motion_confidence": round(motion * 100, 1),
        }

        raw_score = (
            self.WEIGHTS["survivor_count"] * surv_norm
            + self.WEIGHTS["confidence_score"] * conf
            + self.WEIGHTS["flood_risk"] * flood
            + self.WEIGHTS["fire_risk"] * fire
            + self.WEIGHTS["crowd_density"] * crowd
            + self.WEIGHTS["micro_motion_confidence"] * motion
        )

        risk_score = int(round(raw_score * 100))
        
        # --- Extreme Environment Multipliers ---
        # If the environment itself represents an immediate critical threat
        # force a higher baseline risk score regardless of whether victims are visible on camera.
        
        # Flood Extreme Risk
        water_coverage = data.get("water_coverage", 0.0)
        if flood > 0 and water_coverage > 0.30:
            # Add up to 50 extra baseline risk points if water covers > 30% of the screen
            extra_flood_risk = int(min(water_coverage * 100, 50))
            risk_score += extra_flood_risk

        # Fire Extreme Risk
        fire_coverage = data.get("fire_coverage", 0.0)
        if fire > 0:
            # Add up to 60 extra baseline risk points based on fire coverage (fire is highly dangerous)
            extra_fire_risk = int(min((fire_coverage * 150), 60))
            risk_score += extra_fire_risk

        # Heavy Crowd Extreme Risk
        if data.get("is_heavy_crowd_mode", False) and crowd > 0.4:
            # Add up to 60 extra baseline risk points for crushing hazards when mode is active
            extra_crowd_risk = int(min((crowd * 100), 60))
            risk_score += extra_crowd_risk

        risk_score = max(0, min(100, risk_score))

        risk_level = "LOW"
        for threshold, level in self.THRESHOLDS:
            if risk_score >= threshold:
                risk_level = level
                break

        return {
            "risk_level": risk_level,
            "risk_score": risk_score,
            "breakdown": breakdown,
        }
