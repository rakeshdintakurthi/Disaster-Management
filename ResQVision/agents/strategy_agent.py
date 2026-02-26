"""
ResQVision — Strategy Agent
=============================
Rule-based rescue strategy recommendations.
"""


class StrategyAgent:
    """Recommend rescue strategies based on assessed risk level."""

    # Strategy lookup per risk level
    STRATEGIES = {
        "CRITICAL": {
            "strategy": "Immediate full-scale rescue deployment with aerial and ground units",
            "priority": "P0 — IMMEDIATE",
            "equipment": [
                "Search & Rescue K-9 units",
                "Thermal imaging drones",
                "Hydraulic rescue tools (Jaws of Life)",
                "Inflatable rescue boats",
                "Portable medical triage stations",
                "Night-vision equipment",
                "Communication relay towers",
            ],
            "actions": [
                "Activate all available rescue teams",
                "Deploy aerial reconnaissance drones",
                "Establish forward command post",
                "Request military/national guard assistance",
                "Set up emergency medical triage zone",
                "Begin systematic grid search pattern",
            ],
        },
        "HIGH": {
            "strategy": "Rapid response with specialised rescue teams",
            "priority": "P1 — URGENT",
            "equipment": [
                "Rescue boats",
                "Stretchers and spine boards",
                "Portable generators",
                "First-aid kits (advanced)",
                "Thermal cameras",
                "Rope rescue kits",
            ],
            "actions": [
                "Deploy primary rescue team",
                "Launch drone survey of affected area",
                "Prepare medical evacuation routes",
                "Set up temporary shelters nearby",
                "Alert hospital trauma units",
            ],
        },
        "MEDIUM": {
            "strategy": "Targeted search with standby escalation capability",
            "priority": "P2 — HIGH",
            "equipment": [
                "Basic rescue tools",
                "First-aid kits",
                "Megaphones and signal flares",
                "Portable lighting",
                "Life jackets",
            ],
            "actions": [
                "Send scout team for assessment",
                "Prepare rescue teams on standby",
                "Coordinate with local emergency services",
                "Monitor area with drone or camera",
            ],
        },
        "LOW": {
            "strategy": "Monitoring and preventive positioning",
            "priority": "P3 — STANDARD",
            "equipment": [
                "Basic first-aid kit",
                "Communication radios",
                "Area markers / cones",
            ],
            "actions": [
                "Continue surveillance monitoring",
                "Log situation for record",
                "Maintain communication channels open",
            ],
        },
    }

    def recommend(self, risk_level: str) -> dict:
        """
        Generate rescue strategy recommendation.

        Parameters
        ----------
        risk_level : str
            One of LOW / MEDIUM / HIGH / CRITICAL.

        Returns
        -------
        dict  with keys: strategy, priority, equipment, actions
        """
        level = risk_level.upper()
        if level not in self.STRATEGIES:
            level = "LOW"

        return dict(self.STRATEGIES[level])
