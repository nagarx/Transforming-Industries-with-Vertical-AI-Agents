"""
Base configuration for agentic systems.
"""

from pydantic import BaseModel, Field
from typing import Dict, Any, Optional, List, Union
from enum import Enum

class AgentType(str, Enum):
    """Types of agents as defined in the paper."""
    TASK_SPECIFIC = "task_specific"
    MULTI_AGENT = "multi_agent"
    HUMAN_AUGMENTED = "human_augmented"

class AgentPattern(str, Enum):
    """Specific patterns for each agent type."""
    # Task-specific patterns
    RAG_ROUTER = "rag_router"
    REACT_AGENT = "react_agent"
    
    # Multi-agent patterns
    RAG_ORCHESTRATED = "rag_orchestrated"
    COLLABORATIVE_PROBLEM_SOLVER = "collaborative_problem_solver"
    
    # Human-augmented patterns
    HITL_AGENT = "hitl_agent"
    COLLABORATIVE_AGENT = "collaborative_agent"
    SUPERVISORY_AGENT = "supervisory_agent"

class MemoryConfig(BaseModel):
    """Configuration for agent memory."""
    enable_short_term_memory: bool = Field(default=True, description="Enable short-term memory for conversation context")
    enable_long_term_memory: bool = Field(default=False, description="Enable long-term memory for persistent knowledge")
    memory_vector_db: Optional[str] = Field(default="chroma", description="Vector database for storing memory embeddings")
    memory_expiry: Optional[int] = Field(default=None, description="Expiry time for memory entries in seconds")
    max_memory_items: int = Field(default=10, description="Maximum number of memory items to maintain in short-term memory")

class CognitiveSkillConfig(BaseModel):
    """Configuration for cognitive skills."""
    skill_id: str = Field(..., description="Unique identifier for the cognitive skill")
    skill_type: str = Field(..., description="Type of cognitive skill (e.g., risk_assessment, compliance, etc.)")
    model_path: Optional[str] = Field(default=None, description="Path to the model file if locally stored")
    api_endpoint: Optional[str] = Field(default=None, description="API endpoint for remote model inference")
    input_format: Dict[str, Any] = Field(default_factory=dict, description="Expected input format for the skill")
    output_format: Dict[str, Any] = Field(default_factory=dict, description="Expected output format from the skill")

class ToolConfig(BaseModel):
    """Configuration for external tools."""
    tool_id: str = Field(..., description="Unique identifier for the tool")
    tool_type: str = Field(..., description="Type of tool (e.g., api, database, search_engine)")
    description: str = Field(..., description="Description of the tool's functionality")
    api_endpoint: Optional[str] = Field(default=None, description="API endpoint for the tool")
    authentication: Dict[str, Any] = Field(default_factory=dict, description="Authentication details for the tool")
    input_schema: Dict[str, Any] = Field(default_factory=dict, description="Schema for tool inputs")
    output_schema: Dict[str, Any] = Field(default_factory=dict, description="Schema for tool outputs")

class BaseAgentConfig(BaseModel):
    """Base configuration for all agent types."""
    agent_id: str = Field(..., description="Unique identifier for the agent")
    agent_name: str = Field(..., description="Human-readable name for the agent")
    agent_type: AgentType = Field(..., description="Type of agent (task_specific, multi_agent, human_augmented)")
    agent_pattern: AgentPattern = Field(..., description="Specific pattern implementation for the agent")
    description: str = Field(..., description="Description of the agent's functionality")
    
    # Core modules configuration
    model_config_id: str = Field(default="default", description="Identifier for the LLM model configuration")
    memory_config: MemoryConfig = Field(default_factory=MemoryConfig, description="Configuration for agent memory")
    cognitive_skills: List[CognitiveSkillConfig] = Field(default_factory=list, description="List of cognitive skills for the agent")
    tools: List[ToolConfig] = Field(default_factory=list, description="List of tools available to the agent")
    
    # Additional configuration
    max_iterations: int = Field(default=10, description="Maximum number of reasoning iterations")
    timeout_seconds: Optional[int] = Field(default=30, description="Timeout for agent operations in seconds")
    custom_settings: Dict[str, Any] = Field(default_factory=dict, description="Custom agent-specific settings") 