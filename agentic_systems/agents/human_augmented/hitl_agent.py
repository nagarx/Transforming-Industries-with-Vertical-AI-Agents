"""
Human-in-the-Loop (HITL) Agent implementation for agentic systems.

This agent incorporates human feedback into the decision-making process,
allowing for validation, refinement, and oversight of agent operations.
"""

import logging
import time
from typing import Any, Dict, List, Optional, Union, Callable

from .base_human_augmented_agent import BaseHumanAugmentedAgent, HumanFeedback, HumanFeedbackType
from ...core.reasoning import BaseReasoning, ReasoningInput
from ...core.memory import BaseMemory
from ...core.tools import BaseTool
from ...core.cognitive_skills import BaseCognitiveSkill
from ..base_agent import AgentResponse

logger = logging.getLogger(__name__)

class HITLAgent(BaseHumanAugmentedAgent):
    """
    Human-in-the-Loop (HITL) Agent implementation.
    
    This agent operates autonomously to process queries while integrating
    human expertise for validation and refinement of outputs, ensuring
    reliable and context-aware decision-making.
    """
    
    def __init__(
        self,
        agent_id: str,
        name: str,
        description: str,
        reasoning_engine: BaseReasoning,
        human_feedback_fn: Callable[[AgentResponse], HumanFeedback],
        memory: Optional[BaseMemory] = None,
        tools: Optional[List[BaseTool]] = None,
        cognitive_skills: Optional[List[BaseCognitiveSkill]] = None,
        max_iterations: int = 3,
        max_tool_calls: int = 5,
        always_require_feedback: bool = False,
        feedback_timeout: Optional[float] = None,
        confidence_threshold: float = 0.8,
        high_stakes_keywords: Optional[List[str]] = None,
        feedback_required_categories: Optional[List[str]] = None,
    ):
        """
        Initialize Human-in-the-Loop Agent.
        
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
            high_stakes_keywords: Keywords that indicate high-stakes decisions requiring feedback
            feedback_required_categories: Categories of responses always requiring feedback
        """
        super().__init__(
            agent_id=agent_id,
            name=name,
            description=description,
            reasoning_engine=reasoning_engine,
            human_feedback_fn=human_feedback_fn,
            memory=memory,
            tools=tools,
            cognitive_skills=cognitive_skills,
            max_iterations=max_iterations,
            max_tool_calls=max_tool_calls,
            always_require_feedback=always_require_feedback,
            feedback_timeout=feedback_timeout,
            confidence_threshold=confidence_threshold,
        )
        
        self.high_stakes_keywords = high_stakes_keywords or []
        self.feedback_required_categories = feedback_required_categories or []
    
    def execute(self, query: str, context: Optional[Dict[str, Any]] = None) -> AgentResponse:
        """
        Execute the HITL Agent with the given query and context.
        
        Args:
            query: The user query or input to the agent
            context: Additional context for the execution
            
        Returns:
            AgentResponse: The agent's response
        """
        start_time = time.time()
        context = context or {}
        
        # Step 1: Generate initial response
        initial_response = self._generate_initial_response(query, context)
        
        # Step 2: Determine if human feedback is needed
        needs_feedback = self.needs_human_feedback(initial_response, context)
        
        # If feedback is not needed, return the initial response
        if not needs_feedback:
            logger.info(f"HITL Agent {self.agent_id} response does not require human feedback")
            initial_response.metadata["human_feedback_requested"] = False
            initial_response.metadata["execution_time"] = time.time() - start_time
            return initial_response
        
        # Step 3: Request human feedback
        logger.info(f"HITL Agent {self.agent_id} requesting human feedback")
        initial_response.metadata["human_feedback_requested"] = True
        
        feedback = self.get_human_feedback(initial_response, context)
        
        # If no feedback is received (e.g., timeout or error), return the initial response
        if not feedback:
            logger.warning(f"HITL Agent {self.agent_id} did not receive human feedback")
            initial_response.metadata["human_feedback_received"] = False
            initial_response.metadata["execution_time"] = time.time() - start_time
            return initial_response
        
        # Step 4: Apply feedback to generate the final response
        logger.info(f"HITL Agent {self.agent_id} received human feedback: {feedback.feedback_type}")
        final_response = self.apply_feedback(feedback, initial_response, query, context)
        
        # Add execution metadata
        final_response.metadata["human_feedback_received"] = True
        final_response.metadata["execution_time"] = time.time() - start_time
        
        return final_response
    
    def needs_human_feedback(self, response: AgentResponse, context: Dict[str, Any]) -> bool:
        """
        Determine if human feedback is needed for a given response.
        
        Args:
            response: The agent's response
            context: Context of the execution
            
        Returns:
            bool: Whether human feedback is needed
        """
        # Always require feedback if configured to do so
        if self.always_require_feedback:
            return True
        
        # Check if the response confidence is below the threshold
        confidence = response.metadata.get("confidence", 1.0)
        if confidence < self.confidence_threshold:
            logger.info(f"Requesting feedback: Low confidence ({confidence:.2f} < {self.confidence_threshold:.2f})")
            return True
        
        # Check if the response contains high-stakes keywords
        if self.high_stakes_keywords:
            for keyword in self.high_stakes_keywords:
                if keyword.lower() in response.content.lower():
                    logger.info(f"Requesting feedback: High-stakes keyword found: '{keyword}'")
                    return True
        
        # Check if the response category requires feedback
        category = context.get("category") or response.metadata.get("category")
        if category and self.feedback_required_categories:
            if category in self.feedback_required_categories:
                logger.info(f"Requesting feedback: Response category requires feedback: '{category}'")
                return True
        
        # Check if any cognitive skill results indicate the need for feedback
        if response.skill_results:
            for skill_result in response.skill_results:
                skill_name = skill_result.get("skill_name", "")
                skill_output = skill_result.get("result", {})
                
                # Check risk assessment results
                if skill_name == "risk_assessment" and isinstance(skill_output, dict):
                    risk_level = skill_output.get("risk_level")
                    if risk_level in ["high", "medium"]:
                        logger.info(f"Requesting feedback: Risk assessment indicates {risk_level} risk")
                        return True
                
                # Check toxicity detection results
                if skill_name == "toxicity_detection" and isinstance(skill_output, dict):
                    is_toxic = skill_output.get("is_toxic", False)
                    if is_toxic:
                        logger.info("Requesting feedback: Toxicity detected")
                        return True
                
                # Check compliance monitoring results
                if skill_name == "compliance_monitoring" and isinstance(skill_output, dict):
                    is_compliant = skill_output.get("is_compliant", True)
                    if not is_compliant:
                        logger.info("Requesting feedback: Compliance issues detected")
                        return True
        
        # No feedback needed
        return False
    
    def _generate_initial_response(self, query: str, context: Dict[str, Any]) -> AgentResponse:
        """
        Generate the initial response for the query.
        
        Args:
            query: The user query
            context: Additional context
            
        Returns:
            AgentResponse: The initial response
        """
        # Create reasoning input
        reasoning_input = ReasoningInput(
            prompt=query,
            context=[f"Context: {key}={value}" for key, value in context.items()],
            system_prompt=(
                "You are a human-in-the-loop AI assistant. Generate a response to the user's query "
                "that may require human validation before being finalized. Be clear, accurate, and "
                "comprehensive. If you're uncertain about any aspect of your response, explicitly "
                "note areas where human input would be valuable."
            ),
            memory_items=self._get_relevant_memory_items(query),
        )
        
        # Execute reasoning to generate response
        reasoning_output = self.reasoning_engine.reason(reasoning_input)
        
        # Create agent response
        response = AgentResponse(
            content=reasoning_output.response,
            success=True,
            reasoning=reasoning_output.reasoning_trace,
            metadata={
                "confidence": reasoning_output.confidence or 0.7,  # Default to moderate confidence
                "requires_human_feedback": self.always_require_feedback,
            },
        )
        
        # Execute any required cognitive skills for validation
        if self.cognitive_skills:
            skill_results = []
            
            for skill in self.cognitive_skills:
                skill_input = {
                    "text": response.content,
                    "context": query,
                }
                
                try:
                    skill_result = self.execute_skill(skill.name, skill_input)
                    skill_results.append(skill_result)
                except Exception as e:
                    logger.exception(f"Error executing cognitive skill {skill.name}: {str(e)}")
            
            response.skill_results = skill_results
        
        return response
    
    def simulate_human_feedback(self, response: AgentResponse) -> HumanFeedback:
        """
        Simulate human feedback for testing or demonstration purposes.
        
        Args:
            response: The agent's response to give feedback on
            
        Returns:
            HumanFeedback: Simulated human feedback
        """
        # This is a placeholder for simulating different types of feedback
        # In a real system, this would be replaced by actual human feedback
        
        # Simple heuristic: approve responses with high confidence
        confidence = response.metadata.get("confidence", 0.0)
        
        if confidence > 0.9:
            return HumanFeedback(
                feedback_type=HumanFeedbackType.APPROVE,
                content="Approved due to high confidence.",
            )
        elif confidence > 0.7:
            return HumanFeedback(
                feedback_type=HumanFeedbackType.MODIFY,
                content=response.content + "\n\n[Note: This response has been reviewed by a human expert.]",
                modifications={"add_human_validation_note": True},
            )
        else:
            return HumanFeedback(
                feedback_type=HumanFeedbackType.PROVIDE_CONTEXT,
                content="Please provide more specific information and cite your sources.",
                context={"require_citations": True, "be_more_specific": True},
            )
    
    @staticmethod
    def create_console_feedback_fn() -> Callable[[AgentResponse], HumanFeedback]:
        """
        Create a feedback function that requests input from the console.
        
        Returns:
            Callable[[AgentResponse], HumanFeedback]: Function to get feedback from console
        """
        def console_feedback_fn(response: AgentResponse) -> HumanFeedback:
            print("\n" + "="*80)
            print("HUMAN FEEDBACK REQUESTED")
            print("="*80)
            print(f"\nResponse:\n{response.content}\n")
            print("Options:")
            print("1. Approve (a)")
            print("2. Reject (r)")
            print("3. Modify (m)")
            print("4. Provide Context (c)")
            print("5. Clarify (q)")
            print("6. Redirect (d)")
            
            choice = input("\nEnter your choice (a/r/m/c/q/d): ").strip().lower()
            
            if choice in ['1', 'a', 'approve']:
                return HumanFeedback(feedback_type=HumanFeedbackType.APPROVE)
                
            elif choice in ['2', 'r', 'reject']:
                reason = input("Enter reason for rejection: ")
                return HumanFeedback(
                    feedback_type=HumanFeedbackType.REJECT,
                    content=reason,
                )
                
            elif choice in ['3', 'm', 'modify']:
                modified_text = input("Enter modified response (or press Enter to edit): ")
                if not modified_text:
                    # In a real system, you might open an editor here
                    modified_text = response.content + "\n[Modified by human]"
                
                return HumanFeedback(
                    feedback_type=HumanFeedbackType.MODIFY,
                    content=modified_text,
                )
                
            elif choice in ['4', 'c', 'context']:
                additional_context = input("Enter additional context: ")
                return HumanFeedback(
                    feedback_type=HumanFeedbackType.PROVIDE_CONTEXT,
                    content=additional_context,
                    context={"additional_information": additional_context},
                )
                
            elif choice in ['5', 'q', 'clarify']:
                clarification = input("Enter clarification: ")
                return HumanFeedback(
                    feedback_type=HumanFeedbackType.CLARIFY,
                    content=clarification,
                )
                
            elif choice in ['6', 'd', 'redirect']:
                redirect = input("Enter redirection: ")
                return HumanFeedback(
                    feedback_type=HumanFeedbackType.REDIRECT,
                    content=redirect,
                )
                
            else:
                print("Invalid choice, defaulting to approve.")
                return HumanFeedback(feedback_type=HumanFeedbackType.APPROVE)
        
        return console_feedback_fn 