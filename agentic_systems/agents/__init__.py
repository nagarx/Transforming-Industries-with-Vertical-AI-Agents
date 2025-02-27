"""
Agents module for agentic systems.

This module provides various agent implementations based on the paper
"Agentic Systems: A Guide to Transforming Industries with Vertical AI Agents."
"""

from .base_agent import BaseAgent, AgentResponse
from .task_specific import BaseTaskAgent, RagRouterAgent
from .multi_agent import BaseMultiAgentSystem, AgentNode, RagOrchestratedSystem
from .human_augmented import BaseHumanAugmentedAgent, HITLAgent, HumanFeedback, HumanFeedbackType

__all__ = [
    # Base agent
    "BaseAgent",
    "AgentResponse",
    
    # Task-specific agents
    "BaseTaskAgent",
    "RagRouterAgent",
    
    # Multi-agent systems
    "BaseMultiAgentSystem",
    "AgentNode",
    "RagOrchestratedSystem",
    
    # Human-augmented agents
    "BaseHumanAugmentedAgent",
    "HITLAgent",
    "HumanFeedback",
    "HumanFeedbackType",
] 