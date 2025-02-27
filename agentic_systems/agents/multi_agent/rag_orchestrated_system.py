"""
RAG Orchestrated Multi-Agent System implementation for agentic systems.

This system uses a lead orchestrator agent to coordinate specialized agents,
each focused on retrieval tasks from specific knowledge domains or tools.
"""

import logging
import time
from typing import Any, Dict, List, Optional, Union, Tuple

from .base_multi_agent import BaseMultiAgentSystem, AgentNode, AgentResponse
from ...core.reasoning import BaseReasoning, ReasoningInput
from ...core.memory import BaseMemory
from ...core.tools import BaseTool
from ...core.cognitive_skills import BaseCognitiveSkill, RiskAssessmentSkill

logger = logging.getLogger(__name__)

class RagOrchestratedSystem(BaseMultiAgentSystem):
    """
    RAG Orchestrated Multi-Agent System implementation.
    
    This system uses a lead orchestrator agent to dynamically route queries
    to specialized agents, collect their outputs, and integrate the information
    into a unified, context-aware response.
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
        orchestration_strategy: str = "sequential",  # "sequential", "parallel", or "adaptive"
        risk_assessment: Optional[RiskAssessmentSkill] = None,
    ):
        """
        Initialize RAG Orchestrated Multi-Agent System.
        
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
            orchestration_strategy: Strategy for orchestrating agent interactions
            risk_assessment: Risk assessment skill for validating outputs
        """
        # System configuration
        system_config = {
            "orchestration_strategy": orchestration_strategy,
            "use_risk_assessment": risk_assessment is not None,
        }
        
        # Add risk assessment to cognitive skills if provided
        if risk_assessment and system_cognitive_skills:
            if risk_assessment not in system_cognitive_skills:
                system_cognitive_skills.append(risk_assessment)
        elif risk_assessment:
            system_cognitive_skills = [risk_assessment]
        
        super().__init__(
            system_id=system_id,
            name=name,
            description=description,
            agents=agents,
            orchestration_engine=orchestration_engine,
            system_memory=system_memory,
            system_tools=system_tools,
            system_cognitive_skills=system_cognitive_skills,
            max_iterations=max_iterations,
            max_agent_calls=max_agent_calls,
            system_config=system_config,
        )
        
        # Ensure there is a lead agent
        if not self.lead_agent:
            raise ValueError("RAG Orchestrated System requires a lead agent (orchestrator)")
        
        # Store reference to risk assessment skill
        self.risk_assessment = risk_assessment
        self.orchestration_strategy = orchestration_strategy
    
    def execute(self, query: str, context: Optional[Dict[str, Any]] = None) -> AgentResponse:
        """
        Execute the RAG Orchestrated Multi-Agent System with the given query and context.
        
        Args:
            query: The user query to process
            context: Additional context for the execution
            
        Returns:
            AgentResponse: The system's response with integrated information
        """
        start_time = time.time()
        context = context or {}
        
        # Step 1: Orchestrator analyzes the query and determines the execution plan
        execution_plan = self._create_execution_plan(query, context)
        
        # Step 2: Execute the plan by calling specialized agents
        agent_responses = self._execute_plan(execution_plan, query, context)
        
        # Step 3: Integrate results and generate the final response
        final_response = self._integrate_results(query, agent_responses, execution_plan, context)
        
        # Add execution metadata
        final_response.metadata["execution_time"] = time.time() - start_time
        final_response.metadata["execution_plan"] = execution_plan
        final_response.metadata["agent_responses"] = {
            agent_id: response.to_dict() for agent_id, response in agent_responses.items()
        }
        
        # Step 4: Assess risk of the final response if risk assessment is enabled
        if self.risk_assessment:
            risk_assessment_result = self._assess_response_risk(final_response.content, query)
            final_response.metadata["risk_assessment"] = risk_assessment_result
            
            # If high risk is detected, add warning to the response
            if risk_assessment_result.get("risk_level") == "high":
                warning = "\n\nNOTE: This response has been flagged for potential risks: "
                warning += ", ".join(risk_assessment_result.get("high_risk_categories", []))
                warning += ". Please review carefully."
                
                final_response.content += warning
        
        # Update system memory with the interaction
        memory_id = self.update_memory(
            {
                "query": query,
                "response": final_response.content,
                "execution_plan": execution_plan,
                "agent_responses": {
                    agent_id: response.content for agent_id, response in agent_responses.items()
                },
            },
            type="system_execution",
            timestamp=time.time(),
        )
        
        return final_response
    
    def _create_execution_plan(self, query: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create an execution plan for processing the query.
        
        Args:
            query: The user query
            context: Additional context
            
        Returns:
            Dict[str, Any]: Execution plan with tasks for specialized agents
        """
        # Create reasoning input for the orchestrator
        agent_descriptions = "\n".join([
            f"- {agent.agent_id} ({agent.role}): {agent.description}"
            for agent in self.agents if not agent.is_lead
        ])
        
        reasoning_input = ReasoningInput(
            prompt=query,
            context=[
                f"Task: Create an execution plan to answer the following query.",
                f"Available specialized agents:\n{agent_descriptions}",
                f"Orchestration strategy: {self.orchestration_strategy}",
            ],
            system_prompt=(
                f"You are the lead orchestrator in a multi-agent system. "
                f"Your task is to analyze the query and create a plan to divide the work among specialized agents. "
                f"For each agent, specify what specific sub-task or question they should address."
            ),
            memory_items=self._get_relevant_memory_items(query),
        )
        
        # Execute reasoning to create the plan
        reasoning_output = self.lead_agent.agent.reasoning_engine.reason(reasoning_input)
        
        # Parse the reasoning output to extract the execution plan
        plan = self._parse_execution_plan(reasoning_output.response, reasoning_output.next_actions)
        
        # Log the execution plan
        logger.info(f"Execution plan for query '{query[:50]}...': {plan['agent_tasks'].keys()}")
        
        return plan
    
    def _parse_execution_plan(self, plan_text: str, next_actions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Parse the execution plan from the orchestrator's output.
        
        Args:
            plan_text: Text of the execution plan
            next_actions: Structured next actions from the reasoning output
            
        Returns:
            Dict[str, Any]: Parsed execution plan
        """
        # Try to extract a structured plan from next_actions
        agent_tasks = {}
        execution_order = []
        
        # Check for structured actions
        for action in next_actions:
            if action.get("type") == "assign_task":
                agent_id = action.get("agent_id")
                task = action.get("task")
                
                if agent_id and task and agent_id in self.agent_map:
                    agent_tasks[agent_id] = task
                    execution_order.append(agent_id)
        
        # If no structured actions, try to parse from the plan text
        if not agent_tasks:
            # Simple heuristic: look for lines with agent IDs
            for agent in self.agents:
                if not agent.is_lead and agent.agent_id in plan_text:
                    # Find the line containing the agent ID
                    for line in plan_text.split("\n"):
                        if agent.agent_id in line:
                            # Extract the task (everything after the agent ID)
                            task = line.split(agent.agent_id)[-1].strip(": -")
                            agent_tasks[agent.agent_id] = task
                            execution_order.append(agent.agent_id)
                            break
        
        # Determine execution approach
        if self.orchestration_strategy == "sequential":
            parallel_execution = False
        elif self.orchestration_strategy == "parallel":
            parallel_execution = True
        else:  # "adaptive"
            # Use parallel execution if there are multiple independent tasks
            parallel_execution = len(agent_tasks) > 1
        
        return {
            "agent_tasks": agent_tasks,
            "execution_order": execution_order,
            "parallel_execution": parallel_execution,
            "plan_text": plan_text,
        }
    
    def _execute_plan(
        self, 
        execution_plan: Dict[str, Any], 
        query: str, 
        context: Dict[str, Any]
    ) -> Dict[str, AgentResponse]:
        """
        Execute the plan by calling specialized agents.
        
        Args:
            execution_plan: The execution plan
            query: The original user query
            context: Additional context
            
        Returns:
            Dict[str, AgentResponse]: Responses from each agent
        """
        agent_tasks = execution_plan["agent_tasks"]
        execution_order = execution_plan["execution_order"]
        parallel_execution = execution_plan["parallel_execution"]
        
        agent_responses = {}
        
        if parallel_execution:
            # In a real implementation, this would use parallel processing
            # For simplicity, we'll execute sequentially but not build on previous responses
            for agent_id in execution_order:
                task = agent_tasks[agent_id]
                agent_context = context.copy()  # Each agent gets the original context
                
                try:
                    response = self.call_agent(agent_id, task, agent_context)
                    agent_responses[agent_id] = response
                except Exception as e:
                    logger.error(f"Error executing agent {agent_id}: {str(e)}")
                    # Create a failure response
                    agent_responses[agent_id] = AgentResponse(
                        content=f"Failed to execute: {str(e)}",
                        success=False,
                    )
        else:
            # Sequential execution: each agent can build on previous responses
            cumulative_context = context.copy()
            
            for agent_id in execution_order:
                task = agent_tasks[agent_id]
                
                try:
                    response = self.call_agent(agent_id, task, cumulative_context)
                    agent_responses[agent_id] = response
                    
                    # Add this agent's response to the context for subsequent agents
                    cumulative_context[f"response_from_{agent_id}"] = response.content
                except Exception as e:
                    logger.error(f"Error executing agent {agent_id}: {str(e)}")
                    agent_responses[agent_id] = AgentResponse(
                        content=f"Failed to execute: {str(e)}",
                        success=False,
                    )
        
        return agent_responses
    
    def _integrate_results(
        self,
        query: str,
        agent_responses: Dict[str, AgentResponse],
        execution_plan: Dict[str, Any],
        context: Dict[str, Any],
    ) -> AgentResponse:
        """
        Integrate results from specialized agents into a cohesive response.
        
        Args:
            query: The original user query
            agent_responses: Responses from specialized agents
            execution_plan: The execution plan
            context: Additional context
            
        Returns:
            AgentResponse: Integrated response
        """
        # Prepare context for integration
        integration_context = []
        
        # Add each agent's response to the context
        for agent_id, response in agent_responses.items():
            agent_node = self.agent_map[agent_id]
            integration_context.append(
                f"Response from {agent_id} ({agent_node.role}):\n{response.content}"
            )
        
        # Create reasoning input for the orchestrator to integrate results
        reasoning_input = ReasoningInput(
            prompt=query,
            context=integration_context,
            system_prompt=(
                "You are the lead orchestrator integrating responses from specialized agents. "
                "Synthesize the information into a cohesive, comprehensive response that directly "
                "addresses the user's query. Ensure that your response is complete, accurate, and "
                "properly cites information from the appropriate agents when relevant."
            ),
            memory_items=self._get_relevant_memory_items(query),
        )
        
        # Execute reasoning to integrate results
        reasoning_output = self.lead_agent.agent.reasoning_engine.reason(reasoning_input)
        
        # Create the integrated response
        integrated_response = AgentResponse(
            content=reasoning_output.response,
            success=True,
            reasoning=reasoning_output.reasoning_trace,
            tool_calls=[
                {"agent_id": agent_id, "response": response.content}
                for agent_id, response in agent_responses.items()
            ],
            metadata={
                "execution_plan": execution_plan,
                "integration_reasoning": reasoning_output.reasoning_trace,
            },
        )
        
        return integrated_response
    
    def _assess_response_risk(self, response_text: str, query: str) -> Dict[str, Any]:
        """
        Assess the risk level of the response using the risk assessment skill.
        
        Args:
            response_text: The response to assess
            query: The original user query
            
        Returns:
            Dict[str, Any]: Risk assessment result
        """
        if not self.risk_assessment:
            return {"risk_assessed": False}
        
        try:
            # Execute risk assessment skill
            risk_input = {
                "text": response_text,
                "context": f"This is a response to the query: {query}",
            }
            
            risk_result = self.execute_skill("risk_assessment", risk_input)
            
            if risk_result["success"]:
                result_data = risk_result["result"]
                
                # Extract high risk categories
                high_risk_categories = []
                if "categories" in result_data:
                    for category, score in result_data["categories"].items():
                        if score >= self.risk_assessment.risk_threshold:
                            high_risk_categories.append(category)
                
                return {
                    "risk_assessed": True,
                    "risk_level": result_data.get("risk_level", "unknown"),
                    "risk_score": result_data.get("risk_score", 0.0),
                    "high_risk_categories": high_risk_categories,
                    "recommendations": result_data.get("recommendations", []),
                }
            else:
                return {
                    "risk_assessed": True,
                    "risk_level": "unknown",
                    "error": risk_result["error"],
                }
                
        except Exception as e:
            logger.exception(f"Error during risk assessment: {str(e)}")
            return {
                "risk_assessed": True,
                "risk_level": "unknown",
                "error": str(e),
            }
    
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