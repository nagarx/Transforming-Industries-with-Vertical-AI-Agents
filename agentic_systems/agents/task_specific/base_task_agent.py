"""
Base Task-Specific agent interface for agentic systems.

Task-Specific Agents are autonomous systems designed to handle specific functions
or solve narrowly defined problems within particular domains.
"""

from abc import abstractmethod
from typing import Any, Dict, List, Optional, Union
import logging

from ..base_agent import BaseAgent, AgentResponse
from ...core.reasoning import BaseReasoning
from ...core.memory import BaseMemory
from ...core.tools import BaseTool
from ...core.cognitive_skills import BaseCognitiveSkill

logger = logging.getLogger(__name__)

class BaseTaskAgent(BaseAgent):
    """
    Base class for Task-Specific Agents.
    
    Task-Specific Agents are designed to perform specialized functions
    or solve narrowly defined problems. They act as focused modules
    that contribute to larger systems by efficiently managing discrete tasks.
    """
    
    def __init__(
        self,
        agent_id: str,
        name: str,
        description: str,
        reasoning_engine: BaseReasoning,
        task_type: str,
        memory: Optional[BaseMemory] = None,
        tools: Optional[List[BaseTool]] = None,
        cognitive_skills: Optional[List[BaseCognitiveSkill]] = None,
        max_iterations: int = 5,
        max_tool_calls: int = 3,
        task_specific_config: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize Task-Specific agent.
        
        Args:
            agent_id: Unique identifier for the agent
            name: Human-readable name for the agent
            description: Description of the agent's purpose and capabilities
            reasoning_engine: Reasoning engine for the agent
            task_type: Type of task this agent specializes in
            memory: Memory component for the agent
            tools: Tools available to the agent
            cognitive_skills: Cognitive skills available to the agent
            max_iterations: Maximum number of reasoning iterations
            max_tool_calls: Maximum number of tool calls per execution
            task_specific_config: Configuration specific to this task type
        """
        super().__init__(
            agent_id=agent_id,
            name=name,
            description=description,
            reasoning_engine=reasoning_engine,
            memory=memory,
            tools=tools,
            cognitive_skills=cognitive_skills,
            max_iterations=max_iterations,
            max_tool_calls=max_tool_calls,
        )
        
        self.task_type = task_type
        self.task_specific_config = task_specific_config or {}
    
    @abstractmethod
    def execute(self, query: str, context: Optional[Dict[str, Any]] = None) -> AgentResponse:
        """
        Execute the task agent with the given query and context.
        
        Args:
            query: The user query or input to the agent
            context: Additional context for the execution
            
        Returns:
            AgentResponse: The agent's response
        """
        pass
    
    @abstractmethod
    def validate_task_input(self, query: str, context: Optional[Dict[str, Any]] = None) -> bool:
        """
        Validate that the input is appropriate for this task agent.
        
        Args:
            query: The user query or input to the agent
            context: Additional context for the execution
            
        Returns:
            bool: True if the input is valid for this task, False otherwise
        """
        pass
    
    def get_task_info(self) -> Dict[str, Any]:
        """
        Get information about this task agent.
        
        Returns:
            Dict[str, Any]: Information about the task agent
        """
        return {
            "agent_id": self.agent_id,
            "name": self.name,
            "description": self.description,
            "task_type": self.task_type,
            "tools": [tool.name for tool in self.tools],
            "cognitive_skills": [skill.name for skill in self.cognitive_skills],
            "task_specific_config": self.task_specific_config,
        } 