"""
Base Human-Augmented agent interface for agentic systems.

Human-Augmented Agents are intelligent systems designed to collaborate
with humans by automating complex tasks while incorporating human
oversight, feedback, or decision-making.
"""

from abc import abstractmethod
from typing import Any, Dict, List, Optional, Union, Callable
import logging
import time
import enum

from ..base_agent import BaseAgent, AgentResponse
from ...core.reasoning import BaseReasoning
from ...core.memory import BaseMemory
from ...core.tools import BaseTool
from ...core.cognitive_skills import BaseCognitiveSkill

logger = logging.getLogger(__name__)

class HumanFeedbackType(str, enum.Enum):
    """Types of human feedback that can be provided to an agent."""
    
    APPROVE = "approve"
    REJECT = "reject"
    MODIFY = "modify"
    PROVIDE_CONTEXT = "provide_context"
    CLARIFY = "clarify"
    REDIRECT = "redirect"

class HumanFeedback:
    """
    Represents feedback from a human to an agent.
    
    This class encapsulates human feedback, including the decision,
    modifications, and additional context or instructions.
    """
    
    def __init__(
        self,
        feedback_type: HumanFeedbackType,
        content: Optional[str] = None,
        modifications: Optional[Dict[str, Any]] = None,
        context: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize human feedback.
        
        Args:
            feedback_type: Type of feedback
            content: Feedback content or message
            modifications: Specific modifications to make
            context: Additional context to consider
            metadata: Additional metadata about the feedback
        """
        self.feedback_type = feedback_type
        self.content = content or ""
        self.modifications = modifications or {}
        self.context = context or {}
        self.metadata = metadata or {}
        self.timestamp = time.time()
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert feedback to dictionary representation.
        
        Returns:
            Dict[str, Any]: Dictionary representation of the feedback
        """
        return {
            "feedback_type": self.feedback_type,
            "content": self.content,
            "modifications": self.modifications,
            "context": self.context,
            "metadata": self.metadata,
            "timestamp": self.timestamp,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "HumanFeedback":
        """
        Create feedback from dictionary.
        
        Args:
            data: Dictionary with feedback data
            
        Returns:
            HumanFeedback: Feedback instance
        """
        feedback = cls(
            feedback_type=data["feedback_type"],
            content=data.get("content"),
            modifications=data.get("modifications"),
            context=data.get("context"),
            metadata=data.get("metadata"),
        )
        feedback.timestamp = data.get("timestamp", time.time())
        return feedback

class BaseHumanAugmentedAgent(BaseAgent):
    """
    Base class for Human-Augmented Agents.
    
    Human-Augmented Agents incorporate human feedback and oversight
    into their operations, enabling collaboration between AI and humans
    for more reliable and context-aware decision-making.
    """
    
    def __init__(
        self,
        agent_id: str,
        name: str,
        description: str,
        reasoning_engine: BaseReasoning,
        human_feedback_fn: Optional[Callable[[AgentResponse], HumanFeedback]] = None,
        memory: Optional[BaseMemory] = None,
        tools: Optional[List[BaseTool]] = None,
        cognitive_skills: Optional[List[BaseCognitiveSkill]] = None,
        max_iterations: int = 3,
        max_tool_calls: int = 5,
        always_require_feedback: bool = False,
        feedback_timeout: Optional[float] = None,
        confidence_threshold: float = 0.8,
    ):
        """
        Initialize Human-Augmented agent.
        
        Args:
            agent_id: Unique identifier for the agent
            name: Human-readable name for the agent
            description: Description of the agent's purpose and capabilities
            reasoning_engine: Reasoning engine for the agent
            human_feedback_fn: Function to get human feedback
            memory: Memory component for the agent
            tools: Tools available to the agent
            cognitive_skills: Cognitive skills available to the agent
            max_iterations: Maximum number of reasoning iterations
            max_tool_calls: Maximum number of tool calls per execution
            always_require_feedback: Whether to always require human feedback
            feedback_timeout: Timeout for waiting for human feedback
            confidence_threshold: Confidence threshold for requiring feedback
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
        
        self.human_feedback_fn = human_feedback_fn
        self.always_require_feedback = always_require_feedback
        self.feedback_timeout = feedback_timeout
        self.confidence_threshold = confidence_threshold
        
        # Initialize feedback history
        self.feedback_history = []
    
    @abstractmethod
    def execute(self, query: str, context: Optional[Dict[str, Any]] = None) -> AgentResponse:
        """
        Execute the human-augmented agent with the given query and context.
        
        Args:
            query: The user query or input to the agent
            context: Additional context for the execution
            
        Returns:
            AgentResponse: The agent's response
        """
        pass
    
    @abstractmethod
    def needs_human_feedback(self, response: AgentResponse, context: Dict[str, Any]) -> bool:
        """
        Determine if human feedback is needed for a given response.
        
        Args:
            response: The agent's response
            context: Context of the execution
            
        Returns:
            bool: Whether human feedback is needed
        """
        pass
    
    def get_human_feedback(
        self,
        response: AgentResponse,
        context: Optional[Dict[str, Any]] = None,
    ) -> Optional[HumanFeedback]:
        """
        Get feedback from a human on the agent's response.
        
        Args:
            response: The agent's response to get feedback on
            context: Additional context for the feedback
            
        Returns:
            Optional[HumanFeedback]: The human feedback, or None if no feedback is available
        """
        if not self.human_feedback_fn:
            logger.warning("No human feedback function provided.")
            return None
        
        try:
            # Call the human feedback function
            feedback = self.human_feedback_fn(response)
            
            # Record the feedback in history
            self.feedback_history.append(feedback)
            
            # Update memory with the feedback
            self.update_memory(
                feedback.to_dict(),
                type="human_feedback",
                timestamp=feedback.timestamp,
            )
            
            return feedback
            
        except Exception as e:
            logger.exception(f"Error getting human feedback: {str(e)}")
            return None
    
    def apply_feedback(
        self,
        feedback: HumanFeedback,
        response: AgentResponse,
        query: str,
        context: Dict[str, Any],
    ) -> AgentResponse:
        """
        Apply human feedback to a response.
        
        Args:
            feedback: The human feedback to apply
            response: The original response
            query: The original query
            context: The execution context
            
        Returns:
            AgentResponse: The updated response after applying feedback
        """
        start_time = time.time()
        
        if feedback.feedback_type == HumanFeedbackType.APPROVE:
            # No changes needed, just update metadata
            response.metadata["human_feedback"] = "approved"
            response.metadata["human_feedback_time"] = start_time
            return response
            
        elif feedback.feedback_type == HumanFeedbackType.REJECT:
            # Create a new response indicating rejection
            return AgentResponse(
                content="The previous response was rejected." + 
                       (f" Reason: {feedback.content}" if feedback.content else ""),
                success=False,
                reasoning=response.reasoning,
                tool_calls=response.tool_calls,
                skill_results=response.skill_results,
                metadata={
                    **response.metadata,
                    "human_feedback": "rejected",
                    "human_feedback_time": start_time,
                    "human_feedback_content": feedback.content,
                },
            )
            
        elif feedback.feedback_type == HumanFeedbackType.MODIFY:
            # Create a new response with the modifications
            modified_content = feedback.content if feedback.content else response.content
            
            return AgentResponse(
                content=modified_content,
                success=True,
                reasoning=response.reasoning,
                tool_calls=response.tool_calls,
                skill_results=response.skill_results,
                metadata={
                    **response.metadata,
                    "human_feedback": "modified",
                    "human_feedback_time": start_time,
                    "original_content": response.content,
                    "modifications": feedback.modifications,
                },
            )
            
        elif feedback.feedback_type in [HumanFeedbackType.PROVIDE_CONTEXT, HumanFeedbackType.CLARIFY, HumanFeedbackType.REDIRECT]:
            # Regenerate the response with the new context or clarification
            updated_context = context.copy()
            updated_context.update(feedback.context)
            
            # Add the feedback content to the context
            if feedback.content:
                updated_context["human_feedback"] = feedback.content
            
            # Create reasoning input with the feedback
            from ...core.reasoning import ReasoningInput
            
            feedback_prompt = query
            if feedback.feedback_type == HumanFeedbackType.CLARIFY:
                feedback_prompt = f"Clarification: {feedback.content}\nOriginal query: {query}"
            elif feedback.feedback_type == HumanFeedbackType.REDIRECT:
                feedback_prompt = feedback.content or query
            
            reasoning_input = ReasoningInput(
                prompt=feedback_prompt,
                context=[
                    f"Previous response: {response.content}",
                    f"Human feedback: {feedback.content}",
                    f"Feedback type: {feedback.feedback_type.value}",
                ],
                system_prompt=(
                    "You are a human-augmented AI assistant. You received feedback from a human "
                    "on your previous response. Incorporate this feedback to generate an improved "
                    "response that addresses the human's concerns and requirements."
                ),
                memory_items=self._get_relevant_memory_items(query),
            )
            
            # Execute reasoning with the feedback
            reasoning_output = self.reasoning_engine.reason(reasoning_input)
            
            # Create a new response with the refined content
            return AgentResponse(
                content=reasoning_output.response,
                success=True,
                reasoning=reasoning_output.reasoning_trace,
                tool_calls=response.tool_calls,
                skill_results=response.skill_results,
                metadata={
                    **response.metadata,
                    "human_feedback": feedback.feedback_type.value,
                    "human_feedback_time": start_time,
                    "original_content": response.content,
                    "human_feedback_content": feedback.content,
                },
            )
        
        else:
            # Unknown feedback type, return the original response
            logger.warning(f"Unknown feedback type: {feedback.feedback_type}")
            response.metadata["human_feedback"] = "unknown"
            response.metadata["human_feedback_time"] = start_time
            return response
    
    def _get_relevant_memory_items(self, query: str) -> List[Dict[str, Any]]:
        """
        Retrieve relevant memory items for the query.
        
        Args:
            query: The user query
            
        Returns:
            List[Dict[str, Any]]: Relevant memory items
        """
        memory_items = self.retrieve_from_memory(query, limit=3)
        return [item.to_dict() if hasattr(item, "to_dict") else item for item in memory_items] 