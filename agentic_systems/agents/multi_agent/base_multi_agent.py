"""
Base Multi-Agent system interface for agentic systems.

Multi-Agent Systems are collections of autonomous agents designed to
collaborate and solve interconnected problems or achieve shared goals.
"""

from abc import abstractmethod
from typing import Any, Dict, List, Optional, Union
import logging
import uuid
import time

from ..base_agent import BaseAgent, AgentResponse
from ...core.reasoning import BaseReasoning
from ...core.memory import BaseMemory
from ...core.tools import BaseTool
from ...core.cognitive_skills import BaseCognitiveSkill

logger = logging.getLogger(__name__)

class AgentNode:
    """
    Represents an agent within a multi-agent system.
    
    This class wraps an agent and provides additional metadata and
    functionality for integrating it into a multi-agent system.
    """
    
    def __init__(
        self,
        agent: BaseAgent,
        role: str,
        description: str,
        is_lead: bool = False,
        can_communicate_with: Optional[List[str]] = None,
    ):
        """
        Initialize an agent node.
        
        Args:
            agent: The agent instance
            role: The role of this agent in the system
            description: Description of the agent's role and capabilities
            is_lead: Whether this is a lead agent that can coordinate others
            can_communicate_with: List of agent IDs this agent can communicate with
        """
        self.agent = agent
        self.agent_id = agent.agent_id
        self.role = role
        self.description = description
        self.is_lead = is_lead
        self.can_communicate_with = can_communicate_with or []
        
        # Add some metadata for tracking interactions
        self.last_active = None
        self.execution_count = 0
        self.message_history = []
    
    def execute(self, query: str, context: Optional[Dict[str, Any]] = None) -> AgentResponse:
        """
        Execute the agent with the given query and context.
        
        Args:
            query: The query or input to the agent
            context: Additional context for execution
            
        Returns:
            AgentResponse: The agent's response
        """
        start_time = time.time()
        
        # Execute the agent
        response = self.agent.execute(query, context)
        
        # Update metadata
        self.last_active = time.time()
        self.execution_count += 1
        
        # Record the message exchange
        message_record = {
            "timestamp": start_time,
            "query": query,
            "response": response.content,
            "execution_time": time.time() - start_time,
        }
        self.message_history.append(message_record)
        
        return response
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert agent node to dictionary representation.
        
        Returns:
            Dict[str, Any]: Dictionary representation of the agent node
        """
        return {
            "agent_id": self.agent_id,
            "role": self.role,
            "description": self.description,
            "is_lead": self.is_lead,
            "can_communicate_with": self.can_communicate_with,
            "last_active": self.last_active,
            "execution_count": self.execution_count,
        }

class BaseMultiAgentSystem(BaseAgent):
    """
    Base class for multi-agent systems.
    
    Multi-Agent Systems contain multiple agents that collaborate to achieve
    goals or solve problems. This class provides the framework for managing
    these agents and their interactions.
    """
    
    def __init__(
        self,
        system_id: str,
        name: str,
        description: str,
        agents: List[AgentNode],
        orchestration_engine: BaseReasoning,
        system_memory: Optional[BaseMemory] = None,
        system_tools: Optional[List[BaseTool]] = None,
        system_cognitive_skills: Optional[List[BaseCognitiveSkill]] = None,
        max_iterations: int = 10,
        max_agent_calls: int = 5,
        system_config: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize multi-agent system.
        
        Args:
            system_id: Unique identifier for the system
            name: Human-readable name for the system
            description: Description of the system's purpose and capabilities
            agents: List of agent nodes in the system
            orchestration_engine: Reasoning engine for orchestrating agent interactions
            system_memory: Shared memory component for the system
            system_tools: Tools available to the system
            system_cognitive_skills: Cognitive skills available to the system
            max_iterations: Maximum number of reasoning iterations
            max_agent_calls: Maximum number of agent calls per execution
            system_config: Configuration for the multi-agent system
        """
        super().__init__(
            agent_id=system_id,
            name=name,
            description=description,
            reasoning_engine=orchestration_engine,
            memory=system_memory,
            tools=system_tools,
            cognitive_skills=system_cognitive_skills,
            max_iterations=max_iterations,
            max_tool_calls=max_agent_calls,  # Repurposed for agent calls
        )
        
        # Store agent nodes by ID for quick access
        self.agents = agents
        self.agent_map = {agent.agent_id: agent for agent in agents}
        
        # Find the lead agent if any
        self.lead_agent = next((agent for agent in agents if agent.is_lead), None)
        
        # Additional system configuration
        self.system_config = system_config or {}
        
        # Initialize interaction tracking
        self.interaction_history = []
        self.task_id = str(uuid.uuid4())
    
    @abstractmethod
    def execute(self, query: str, context: Optional[Dict[str, Any]] = None) -> AgentResponse:
        """
        Execute the multi-agent system with the given query and context.
        
        Args:
            query: The user query or input to the system
            context: Additional context for the execution
            
        Returns:
            AgentResponse: The system's response
        """
        pass
    
    def call_agent(self, agent_id: str, query: str, context: Optional[Dict[str, Any]] = None) -> AgentResponse:
        """
        Call a specific agent in the system.
        
        Args:
            agent_id: ID of the agent to call
            query: Query or instruction for the agent
            context: Additional context for the agent
            
        Returns:
            AgentResponse: The agent's response
            
        Raises:
            ValueError: If the agent is not found
        """
        if agent_id not in self.agent_map:
            raise ValueError(f"Agent '{agent_id}' not found in the system")
        
        agent_node = self.agent_map[agent_id]
        
        # Record the intent to call this agent
        call_record = {
            "timestamp": time.time(),
            "caller": "system",
            "callee": agent_id,
            "query": query,
        }
        self.interaction_history.append(call_record)
        
        # Execute the agent
        response = agent_node.execute(query, context)
        
        # Update the record with the response
        call_record["response"] = response.content
        call_record["execution_time"] = time.time() - call_record["timestamp"]
        
        return response
    
    def agent_to_agent_call(
        self,
        caller_id: str,
        callee_id: str,
        query: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> Optional[AgentResponse]:
        """
        Facilitate a call from one agent to another.
        
        Args:
            caller_id: ID of the calling agent
            callee_id: ID of the agent being called
            query: Query or instruction for the callee
            context: Additional context for the callee
            
        Returns:
            Optional[AgentResponse]: The callee's response, or None if the call is not allowed
        """
        # Verify that the caller and callee exist
        if caller_id not in self.agent_map or callee_id not in self.agent_map:
            logger.warning(f"Agent-to-agent call failed: Invalid agent ID ({caller_id} -> {callee_id})")
            return None
        
        caller = self.agent_map[caller_id]
        callee = self.agent_map[callee_id]
        
        # Check if the caller is allowed to communicate with the callee
        if not caller.is_lead and callee_id not in caller.can_communicate_with:
            logger.warning(f"Agent-to-agent call blocked: {caller_id} is not allowed to call {callee_id}")
            return None
        
        # Record the agent-to-agent call
        call_record = {
            "timestamp": time.time(),
            "caller": caller_id,
            "callee": callee_id,
            "query": query,
        }
        self.interaction_history.append(call_record)
        
        # Execute the callee agent
        response = callee.execute(query, context)
        
        # Update the record with the response
        call_record["response"] = response.content
        call_record["execution_time"] = time.time() - call_record["timestamp"]
        
        return response
    
    def get_agent_info(self, agent_id: str) -> Optional[Dict[str, Any]]:
        """
        Get information about a specific agent in the system.
        
        Args:
            agent_id: ID of the agent
            
        Returns:
            Optional[Dict[str, Any]]: Information about the agent, or None if not found
        """
        if agent_id not in self.agent_map:
            return None
        
        return self.agent_map[agent_id].to_dict()
    
    def get_all_agents_info(self) -> List[Dict[str, Any]]:
        """
        Get information about all agents in the system.
        
        Returns:
            List[Dict[str, Any]]: Information about all agents
        """
        return [agent.to_dict() for agent in self.agents]
    
    def get_interaction_history(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Get the interaction history of the multi-agent system.
        
        Args:
            limit: Maximum number of interactions to return
            
        Returns:
            List[Dict[str, Any]]: Interaction history
        """
        if limit is not None:
            return self.interaction_history[-limit:]
        return self.interaction_history 