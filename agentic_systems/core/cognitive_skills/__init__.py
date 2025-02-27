"""
Cognitive skills module for agentic systems.

This module provides specialized inference capabilities for LLM agents,
bridging the gap between general LLM capabilities and domain-specific tasks.
"""

from .base_skill import BaseCognitiveSkill
from .risk_assessment import RiskAssessmentSkill
from .toxicity_detection import ToxicityDetectionSkill
from .compliance_monitoring import ComplianceMonitoringSkill

__all__ = [
    "BaseCognitiveSkill",
    "RiskAssessmentSkill",
    "ToxicityDetectionSkill",
    "ComplianceMonitoringSkill",
] 