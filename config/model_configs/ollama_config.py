"""
Configuration for Ollama models used in the agentic system.
"""

from pydantic import BaseModel, Field
from typing import Dict, Any, Optional, List

class OllamaModelConfig(BaseModel):
    """Configuration for an Ollama model."""
    model_name: str = Field(default="deepseek-r1:14b", description="Name of the Ollama model to use")
    base_url: str = Field(default="http://localhost:11434", description="Base URL for Ollama API")
    
    # Generation parameters
    temperature: float = Field(default=0.7, description="Temperature for model sampling")
    max_tokens: int = Field(default=1024, description="Maximum number of tokens to generate")
    top_p: float = Field(default=0.9, description="Top-p sampling parameter")
    top_k: int = Field(default=40, description="Top-k sampling parameter")
    
    # System prompts for different agent roles
    system_prompts: Dict[str, str] = Field(
        default={
            "default": "You are an AI assistant built to help users with their questions and tasks.",
            "router": "You are a specialized routing agent that determines which knowledge domain or tool is most appropriate for a given query.",
            "orchestrator": "You are an orchestration agent responsible for breaking down complex tasks and coordinating multiple specialized agents.",
            "hitl": "You are a collaborative agent working with human experts to provide recommendations for review and approval."
        },
        description="System prompts for different agent roles"
    )
    
    def get_generation_params(self) -> Dict[str, Any]:
        """Get parameters for generation requests to Ollama."""
        return {
            "model": self.model_name,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "top_p": self.top_p,
            "top_k": self.top_k,
        }
    
    def get_system_prompt(self, role: str) -> str:
        """Get the system prompt for a specific agent role."""
        return self.system_prompts.get(role, self.system_prompts["default"])


# Default configuration for the deepseek-r1:14b model
DEFAULT_MODEL_CONFIG = OllamaModelConfig()

# Configuration optimized for routing tasks (lower temperature for more deterministic outputs)
ROUTER_MODEL_CONFIG = OllamaModelConfig(
    temperature=0.3,
    max_tokens=256,
)

# Configuration optimized for orchestration (balanced parameters)
ORCHESTRATOR_MODEL_CONFIG = OllamaModelConfig(
    temperature=0.5,
    max_tokens=2048,
)

# Configuration optimized for human-in-the-loop scenarios (higher max_tokens for detailed explanations)
HITL_MODEL_CONFIG = OllamaModelConfig(
    temperature=0.6,
    max_tokens=4096,
) 