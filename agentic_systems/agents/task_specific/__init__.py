"""
Task-Specific Agents module for agentic systems.

This module provides implementations of Task-Specific Agents, which are
autonomous systems designed to handle specific functions or solve narrowly
defined problems within particular domains.
"""

from .base_task_agent import BaseTaskAgent
from .rag_router_agent import RagRouterAgent

__all__ = [
    "BaseTaskAgent",
    "RagRouterAgent",
] 