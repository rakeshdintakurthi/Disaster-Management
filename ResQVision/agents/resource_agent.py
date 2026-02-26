"""
ResQVision — Resource Allocation Agent
========================================
Rule-based emergency resource calculator.
"""

import math


class ResourceAgent:
    """Calculate required emergency resources based on risk and survivor count."""

    # Risk-level multipliers
    MULTIPLIERS = {
        "CRITICAL": 2.0,
        "HIGH": 1.5,
        "MEDIUM": 1.0,
        "LOW": 0.5,
    }

    def allocate(self, risk_level: str, survivor_count: int, crowd_density: float = 0.0) -> dict:
        """
        Calculate resource requirements.

        Parameters
        ----------
        risk_level : str
            One of LOW / MEDIUM / HIGH / CRITICAL.
        survivor_count : int
            Number of confirmed survivors.

        Returns
        -------
        dict
            {
                "ambulances": int,
                "rescue_teams": int,
                "medical_units": int,
                "helicopters": int,
                "supply_trucks": int,
                "total_personnel": int,
                "risk_multiplier": float,
            }
        """
        level = risk_level.upper()
        mult = self.MULTIPLIERS.get(level, 1.0)
        survivors = max(survivor_count, 1)

        ambulances = math.ceil(survivors / 3 * mult)
        rescue_teams = math.ceil(survivors / 2 * mult)
        medical_units = math.ceil(survivors / 4 * mult)
        helicopters = math.ceil(survivors / 8 * mult) if level in ("CRITICAL", "HIGH") else 0
        supply_trucks = math.ceil(survivors / 6 * mult)
        
        # Autonomous Crowd Management Units
        # Scale based on 0.0-1.0 crowd density ratio
        police_units = math.ceil(crowd_density * 8 * mult) if crowd_density > 0.3 else 0
        crowd_control_staff = math.ceil(crowd_density * 15 * mult) if crowd_density > 0.5 else 0

        # Personnel estimate: ~5 per rescue team + 3 per medical unit + 2 per ambulance 
        # Add police officers (2 per unit) and crowd staff (1 per staff)
        total_personnel = (rescue_teams * 5) + (medical_units * 3) + (ambulances * 2) + (police_units * 2) + crowd_control_staff

        return {
            "ambulances": ambulances,
            "rescue_teams": rescue_teams,
            "medical_units": medical_units,
            "helicopters": helicopters,
            "supply_trucks": supply_trucks,
            "police_units": police_units,
            "crowd_control_staff": crowd_control_staff,
            "total_personnel": total_personnel,
            "risk_multiplier": mult,
        }
