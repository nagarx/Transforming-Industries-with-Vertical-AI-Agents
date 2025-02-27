"""
Base agent interface for agentic systems.

This module provides the foundation for all agent implementations,
defining common interfaces and behaviors.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union, Tuple
import logging
import time
import uuid

from ..core.memory import BaseMemory, ShortTermMemory
from ..core.reasoning import BaseReasoning
from ..core.tools import BaseTool
from ..core.cognitive_skills import BaseCognitiveSkill

logger = logging.getLogger(__name__)

class AgentResponse:
    """Response from an agent execution."""
    
    def __init__(
        self,
        content: str,
        success: bool = True,
        reasoning: Optional[str] = None,
        tool_calls: Optional[List[Dict[str, Any]]] = None,
        skill_results: Optional[List[Dict[str, Any]]] = None,
        memory_updates: Optional[List[Dict[str, Any]]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize agent response.
        
        Args:
            content: The main response content
            success: Whether the execution was successful
            reasoning: The reasoning process behind the response
            tool_calls: Details of tool calls made during execution
            skill_results: Results from cognitive skills used
            memory_updates: Updates made to the agent's memory
            metadata: Additional metadata about the execution
        """
        self.content = content
        self.success = success
        self.reasoning = reasoning
        self.tool_calls = tool_calls or []
        self.skill_results = skill_results or []
        self.memory_updates = memory_updates or []
        self.metadata = metadata or {}
        self.timestamp = time.time()
        self.id = str(uuid.uuid4())
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert agent response to dictionary.
        
        Returns:
            Dict[str, Any]: Dictionary representation of the agent response
        """
        return {
            "id": self.id,
            "content": self.content,
            "success": self.success,
            "reasoning": self.reasoning,
            "tool_calls": self.tool_calls,
            "skill_results": self.skill_results,
            "memory_updates": self.memory_updates,
            "metadata": self.metadata,
            "timestamp": self.timestamp,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AgentResponse":
        """
        Create agent response from dictionary.
        
        Args:
            data: Dictionary with response data
            
        Returns:
            AgentResponse: Agent response instance
        """
        response = cls(
            content=data["content"],
            success=data["success"],
            reasoning=data.get("reasoning"),
            tool_calls=data.get("tool_calls", []),
            skill_results=data.get("skill_results", []),
            memory_updates=data.get("memory_updates", []),
            metadata=data.get("metadata", {}),
        )
        response.timestamp = data.get("timestamp", time.time())
        response.id = data.get("id", str(uuid.uuid4()))
        return response

class BaseAgent(ABC):
    """
    Abstract base class for all agent implementations.
    
    This class defines the common interface and behaviors for all
    types of agents in the agentic systems framework.
    """
    
    def __init__(
        self,
        agent_id: str,
        name: str,
        description: str,
        reasoning_engine: BaseReasoning,
        memory: Optional[BaseMemory] = None,
        tools: Optional[List[BaseTool]] = None,
        cognitive_skills: Optional[List[BaseCognitiveSkill]] = None,
        max_iterations: int = 10,
        max_tool_calls: int = 5,
    ):
        """
        Initialize base agent.
        
        Args:
            agent_id: Unique identifier for the agent
            name: Human-readable name for the agent
            description: Description of the agent's purpose and capabilities
            reasoning_engine: Reasoning engine for the agent
            memory: Memory component for the agent
            tools: Tools available to the agent
            cognitive_skills: Cognitive skills available to the agent
            max_iterations: Maximum number of reasoning iterations
            max_tool_calls: Maximum number of tool calls per execution
        """
        self.agent_id = agent_id
        self.name = name
        self.description = description
        self.reasoning_engine = reasoning_engine
        self.memory = memory or ShortTermMemory()
        self.tools = tools or []
        self.cognitive_skills = cognitive_skills or []
        self.max_iterations = max_iterations
        self.max_tool_calls = max_tool_calls
        
        # Dictionary mapping tool names to tool objects for quick lookup
        self.tool_map = {tool.name: tool for tool in self.tools}
        
        # Dictionary mapping skill names to skill objects for quick lookup
        self.skill_map = {skill.name: skill for skill in self.cognitive_skills}
        
        # Initialize state
        self.conversation_id = str(uuid.uuid4())
        self.last_execution_time = None
        self.execution_count = 0
    
    @abstractmethod
    def execute(self, query: str, context: Optional[Dict[str, Any]] = None) -> AgentResponse:
        """
        Execute the agent with the given query and context.
        
        Args:
            query: The user query or input to the agent
            context: Additional context for the execution
            
        Returns:
            AgentResponse: The agent's response
        """
        pass
    
    def execute_tool(self, tool_name: str, **kwargs) -> Dict[str, Any]:
        """
        Execute a tool by name with the given parameters.
        
        Args:
            tool_name: Name of the tool to execute
            **kwargs: Parameters for the tool
            
        Returns:
            Dict[str, Any]: Result of the tool execution
            
        Raises:
            ValueError: If the tool is not found
        """
        if tool_name not in self.tool_map:
            raise ValueError(f"Tool '{tool_name}' not found")
        
        tool = self.tool_map[tool_name]
        start_time = time.time()
        
        try:
            response = tool.execute(**kwargs)
            
            result = {
                "tool_name": tool_name,
                "success": response.success,
                "result": response.result,
                "error": response.error,
                "execution_time": time.time() - start_time,
            }
            
            # Log the tool execution
            if response.success:
                logger.info(f"Tool '{tool_name}' executed successfully in {result['execution_time']:.2f}s")
            else:
                logger.warning(f"Tool '{tool_name}' execution failed: {response.error}")
            
            return result
            
        except Exception as e:
            error_msg = f"Error executing tool '{tool_name}': {str(e)}"
            logger.exception(error_msg)
            
            return {
                "tool_name": tool_name,
                "success": False,
                "result": None,
                "error": error_msg,
                "execution_time": time.time() - start_time,
            }
    
    def execute_skill(self, skill_name: str, input_data: Any) -> Dict[str, Any]:
        """
        Execute a cognitive skill by name with the given input.
        
        Args:
            skill_name: Name of the skill to execute
            input_data: Input data for the skill
            
        Returns:
            Dict[str, Any]: Result of the skill execution
            
        Raises:
            ValueError: If the skill is not found
        """
        if skill_name not in self.skill_map:
            raise ValueError(f"Cognitive skill '{skill_name}' not found")
        
        skill = self.skill_map[skill_name]
        start_time = time.time()
        
        try:
            response = skill.execute(input_data)
            
            result = {
                "skill_name": skill_name,
                "success": response.success,
                "result": response.result,
                "confidence": response.confidence,
                "error": response.error,
                "execution_time": time.time() - start_time,
            }
            
            # Log the skill execution
            if response.success:
                logger.info(f"Skill '{skill_name}' executed successfully in {result['execution_time']:.2f}s")
            else:
                logger.warning(f"Skill '{skill_name}' execution failed: {response.error}")
            
            return result
            
        except Exception as e:
            error_msg = f"Error executing skill '{skill_name}': {str(e)}"
            logger.exception(error_msg)
            
            return {
                "skill_name": skill_name,
                "success": False,
                "result": None,
                "confidence": None,
                "error": error_msg,
                "execution_time": time.time() - start_time,
            }
    
    def update_memory(self, item: Any, **metadata) -> str:
        """
        Add an item to the agent's memory.
        
        Args:
            item: The item to add to memory
            **metadata: Additional metadata for the memory item
            
        Returns:
            str: ID of the added memory item
        """
        if not hasattr(self.memory, 'add') or not callable(self.memory.add):
            logger.warning("Memory component does not support adding items")
            return ""
        
        return self.memory.add(item, **metadata)
    
    def retrieve_from_memory(self, query: Any, limit: int = 5, **filters) -> List[Any]:
        """
        Retrieve items from the agent's memory.
        
        Args:
            query: Query to search for in memory
            limit: Maximum number of items to retrieve
            **filters: Additional filters for memory retrieval
            
        Returns:
            List[Any]: Retrieved memory items
        """
        if not hasattr(self.memory, 'search') or not callable(self.memory.search):
            logger.warning("Memory component does not support searching")
            return []
        
        try:
            return self.memory.search(query, limit, **filters)
        except Exception as e:
            logger.exception(f"Error retrieving from memory: {str(e)}")
            return []
    
    def get_available_tools(self) -> List[Dict[str, Any]]:
        """
        Get information about available tools.
        
        Returns:
            List[Dict[str, Any]]: Information about available tools
        """
        return [tool.get_schema() for tool in self.tools]
    
    def get_available_skills(self) -> List[Dict[str, Any]]:
        """
        Get information about available cognitive skills.
        
        Returns:
            List[Dict[str, Any]]: Information about available cognitive skills
        """
        return [skill.get_schema() for skill in self.cognitive_skills]
    
    def add_tool(self, tool: BaseTool) -> None:
        """
        Add a tool to the agent.
        
        Args:
            tool: Tool to add
        """
        self.tools.append(tool)
        self.tool_map[tool.name] = tool
    
    def add_cognitive_skill(self, skill: BaseCognitiveSkill) -> None:
        """
        Add a cognitive skill to the agent.
        
        Args:
            skill: Cognitive skill to add
        """
        self.cognitive_skills.append(skill)
        self.skill_map[skill.name] = skill
    
    def set_reasoning_engine(self, reasoning_engine: BaseReasoning) -> None:
        """
        Set the reasoning engine for the agent.
        
        Args:
            reasoning_engine: New reasoning engine
        """
        self.reasoning_engine = reasoning_engine
    
    def set_memory(self, memory: BaseMemory) -> None:
        """
        Set the memory component for the agent.
        
        Args:
            memory: New memory component
        """
        self.memory = memory 