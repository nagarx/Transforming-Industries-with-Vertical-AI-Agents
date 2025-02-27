"""
Base reasoning interface for agentic systems.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union

class ReasoningInput:
    """Input to the reasoning engine."""
    
    def __init__(
        self,
        prompt: str,
        context: Optional[List[str]] = None,
        system_prompt: Optional[str] = None,
        memory_items: Optional[List[Dict[str, Any]]] = None,
        tool_results: Optional[List[Dict[str, Any]]] = None,
        cognitive_skill_results: Optional[List[Dict[str, Any]]] = None,
        persona: Optional[str] = None,
    ):
        """
        Initialize reasoning input.
        
        Args:
            prompt: The user prompt or query
            context: Additional context for the reasoning
            system_prompt: System prompt to use
            memory_items: Items from memory to include
            tool_results: Results from tool calls
            cognitive_skill_results: Results from cognitive skills
            persona: Persona to adopt for reasoning
        """
        self.prompt = prompt
        self.context = context or []
        self.system_prompt = system_prompt
        self.memory_items = memory_items or []
        self.tool_results = tool_results or []
        self.cognitive_skill_results = cognitive_skill_results or []
        self.persona = persona

class ReasoningOutput:
    """Output from the reasoning engine."""
    
    def __init__(
        self,
        response: str,
        reasoning_trace: Optional[str] = None,
        next_actions: Optional[List[Dict[str, Any]]] = None,
        confidence: Optional[float] = None,
        memory_updates: Optional[List[Dict[str, Any]]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize reasoning output.
        
        Args:
            response: The response to the user
            reasoning_trace: Trace of the reasoning process
            next_actions: Actions to take next
            confidence: Confidence score for the response
            memory_updates: Updates to make to memory
            metadata: Additional metadata
        """
        self.response = response
        self.reasoning_trace = reasoning_trace
        self.next_actions = next_actions or []
        self.confidence = confidence
        self.memory_updates = memory_updates or []
        self.metadata = metadata or {}

class BaseReasoning(ABC):
    """Abstract base class for reasoning implementations."""
    
    @abstractmethod
    def reason(self, input_data: ReasoningInput) -> ReasoningOutput:
        """
        Perform reasoning based on input.
        
        Args:
            input_data: Input data for reasoning
            
        Returns:
            ReasoningOutput: The reasoning output
        """
        pass
    
    @abstractmethod
    def generate_system_prompt(self, persona: Optional[str] = None, context: Optional[str] = None) -> str:
        """
        Generate a system prompt for the reasoning engine.
        
        Args:
            persona: Persona to adopt for reasoning
            context: Additional context to include
            
        Returns:
            str: The generated system prompt
        """
        pass
    
    @abstractmethod
    def parse_response(self, response: str) -> Dict[str, Any]:
        """
        Parse a response from the LLM.
        
        Args:
            response: The raw response
            
        Returns:
            Dict[str, Any]: The parsed response
        """
        pass
    
    @abstractmethod
    def format_prompt(self, input_data: ReasoningInput) -> str:
        """
        Format a prompt for the LLM.
        
        Args:
            input_data: Input data for reasoning
            
        Returns:
            str: The formatted prompt
        """
        pass 