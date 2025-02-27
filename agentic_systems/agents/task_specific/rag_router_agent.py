"""
RAG Router Agent implementation for agentic systems.

The RAG Router Agent dynamically orchestrates knowledge retrieval in
Retrieval-Augmented Generation (RAG) systems by mapping queries to
appropriate domain-specific knowledge sources, tools, or APIs.
"""

import logging
import time
from typing import Any, Dict, List, Optional, Union, Tuple

from .base_task_agent import BaseTaskAgent, AgentResponse
from ...core.reasoning import BaseReasoning, ReasoningInput
from ...core.memory import BaseMemory
from ...core.tools import BaseTool, VectorSearchTool
from ...core.cognitive_skills import BaseCognitiveSkill
from ...core.reasoning.base_reasoning import ReasoningOutput

logger = logging.getLogger(__name__)

class RagRouterAgent(BaseTaskAgent):
    """
    RAG Router Agent implementation.
    
    This agent analyzes user queries and maps them to the appropriate
    domain-specific knowledge sources or tools, enabling efficient 
    retrieval augmented generation across multiple domains.
    """
    
    def __init__(
        self,
        agent_id: str,
        name: str,
        description: str,
        reasoning_engine: BaseReasoning,
        vector_search_tools: List[VectorSearchTool],
        memory: Optional[BaseMemory] = None,
        additional_tools: Optional[List[BaseTool]] = None,
        cognitive_skills: Optional[List[BaseCognitiveSkill]] = None,
        max_iterations: int = 5,
        max_tool_calls: int = 3,
        confidence_threshold: float = 0.7,
        combine_results: bool = False,
        default_search_tool: Optional[str] = None,
    ):
        """
        Initialize RAG Router Agent.
        
        Args:
            agent_id: Unique identifier for the agent
            name: Human-readable name for the agent
            description: Description of the agent's purpose and capabilities
            reasoning_engine: Reasoning engine for the agent
            vector_search_tools: Vector search tools representing different knowledge domains
            memory: Memory component for the agent
            additional_tools: Additional tools available to the agent
            cognitive_skills: Cognitive skills available to the agent
            max_iterations: Maximum number of reasoning iterations
            max_tool_calls: Maximum number of tool calls per execution
            confidence_threshold: Confidence threshold for routing decisions
            combine_results: Whether to combine results from multiple knowledge sources
            default_search_tool: Default search tool to use if routing is uncertain
        """
        # Combine vector search tools with additional tools
        all_tools = vector_search_tools.copy()
        if additional_tools:
            all_tools.extend(additional_tools)
        
        # Initialize with task-specific configuration
        task_specific_config = {
            "confidence_threshold": confidence_threshold,
            "combine_results": combine_results,
            "default_search_tool": default_search_tool,
            "knowledge_domains": {tool.name: tool.description for tool in vector_search_tools},
        }
        
        super().__init__(
            agent_id=agent_id,
            name=name,
            description=description,
            reasoning_engine=reasoning_engine,
            task_type="rag_router",
            memory=memory,
            tools=all_tools,
            cognitive_skills=cognitive_skills,
            max_iterations=max_iterations,
            max_tool_calls=max_tool_calls,
            task_specific_config=task_specific_config,
        )
        
        # Store references to vector search tools for quick access
        self.vector_search_tools = vector_search_tools
        self.vector_tool_map = {tool.name: tool for tool in vector_search_tools}
        self.confidence_threshold = confidence_threshold
        self.combine_results = combine_results
        self.default_search_tool = default_search_tool
    
    def execute(self, query: str, context: Optional[Dict[str, Any]] = None) -> AgentResponse:
        """
        Execute the RAG Router Agent with the given query and context.
        
        Args:
            query: The user query to route to appropriate knowledge domains
            context: Additional context for the execution
            
        Returns:
            AgentResponse: The agent's response with retrieved information
        """
        start_time = time.time()
        context = context or {}
        
        # Validate input
        if not self.validate_task_input(query, context):
            return AgentResponse(
                content="Invalid input for RAG routing. Please provide a clear query.",
                success=False,
                reasoning="Query validation failed.",
                metadata={"execution_time": time.time() - start_time},
            )
        
        # Step 1: Determine which knowledge domain(s) to route to
        routing_decision = self._make_routing_decision(query, context)
        
        # Step 2: Execute vector search on selected domain(s)
        search_results = self._execute_vector_searches(query, routing_decision)
        
        # Step 3: Generate response based on search results
        response = self._generate_response(query, search_results, routing_decision, context)
        
        # Add execution metadata
        response.metadata["execution_time"] = time.time() - start_time
        response.metadata["routing_decision"] = routing_decision
        
        # Update memory with query and response
        memory_id = self.update_memory(
            {
                "query": query,
                "response": response.content,
                "routing_decision": routing_decision,
            },
            type="interaction",
            timestamp=time.time(),
        )
        
        return response
    
    def validate_task_input(self, query: str, context: Optional[Dict[str, Any]] = None) -> bool:
        """
        Validate that the input is appropriate for the RAG Router Agent.
        
        Args:
            query: The user query or input to the agent
            context: Additional context for the execution
            
        Returns:
            bool: True if the input is valid for this task, False otherwise
        """
        # Basic validation: check if query is non-empty
        if not query or not query.strip():
            return False
        
        # Check if there are vector search tools available
        if not self.vector_search_tools:
            logger.error("No vector search tools available for RAG routing")
            return False
        
        return True
    
    def _make_routing_decision(self, query: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Determine which knowledge domain(s) to route the query to.
        
        Args:
            query: The user query
            context: Additional context
            
        Returns:
            Dict[str, Any]: Routing decision with selected domains and confidence scores
        """
        # Prepare knowledge domain information for the reasoning engine
        knowledge_domains = [
            {
                "name": tool.name,
                "description": tool.description,
                "domain": tool.name.split("_")[0] if "_" in tool.name else tool.name,
            }
            for tool in self.vector_search_tools
        ]
        
        # Create reasoning input with routing task
        reasoning_input = ReasoningInput(
            prompt=query,
            context=[
                f"Task: Route the following query to the most appropriate knowledge domain(s).",
                f"Available knowledge domains: {[domain['name'] for domain in knowledge_domains]}",
                f"Knowledge domain descriptions: {[(domain['name'], domain['description']) for domain in knowledge_domains]}",
                f"Confidence threshold: {self.confidence_threshold}",
                f"Combine results: {self.combine_results}",
            ],
            system_prompt=(
                f"You are a routing agent that determines which knowledge domain is most "
                f"appropriate for a given query. Analyze the query and select the domain(s) "
                f"that best match the query's intent and content. "
                f"If confidence is below {self.confidence_threshold} for all domains, "
                f"{'select multiple domains.' if self.combine_results else f'use the default domain: {self.default_search_tool}.'}"
            ),
            memory_items=self._get_relevant_memory_items(query),
        )
        
        # Execute reasoning to make routing decision
        reasoning_output = self.reasoning_engine.reason(reasoning_input)
        
        # Parse the reasoning output to determine routing
        routing_result = self._parse_routing_decision(reasoning_output, knowledge_domains)
        
        # Log the routing decision
        logger.info(f"Routing decision for query '{query[:50]}...': {routing_result['selected_domains']}")
        
        return routing_result
    
    def _parse_routing_decision(self, reasoning_output: ReasoningOutput, knowledge_domains: List[Dict[str, str]]) -> Dict[str, Any]:
        """
        Parse the routing decision from the reasoning output.
        
        Args:
            reasoning_output: Output from the reasoning engine
            knowledge_domains: List of available knowledge domains
            
        Returns:
            Dict[str, Any]: Parsed routing decision
        """
        # Extract domain selections from reasoning output
        selected_domains = []
        domain_confidences = {}
        
        # Check for structured next_actions in the reasoning output
        for action in reasoning_output.next_actions:
            if action.get("type") == "select_domain":
                domain = action.get("domain")
                confidence = action.get("confidence", 0.0)
                
                if domain and domain in self.vector_tool_map:
                    selected_domains.append(domain)
                    domain_confidences[domain] = confidence
        
        # If no structured actions, try to parse from the response text
        if not selected_domains:
            response_text = reasoning_output.response.lower()
            
            for domain in self.vector_tool_map:
                if domain.lower() in response_text:
                    # Simple heuristic to estimate confidence
                    confidence = 0.8  # Default confidence if mentioned explicitly
                    selected_domains.append(domain)
                    domain_confidences[domain] = confidence
        
        # If still no domains selected or all below threshold, use default or combine
        if not selected_domains or all(conf < self.confidence_threshold for conf in domain_confidences.values()):
            if self.combine_results:
                # Use all domains with confidence proportional to specificity
                for i, domain in enumerate(self.vector_tool_map):
                    if domain not in selected_domains:
                        selected_domains.append(domain)
                        # Assign lower confidence to domains not explicitly selected
                        domain_confidences[domain] = 0.5
            elif self.default_search_tool and self.default_search_tool in self.vector_tool_map:
                # Use default search tool
                selected_domains = [self.default_search_tool]
                domain_confidences[self.default_search_tool] = 0.6
            else:
                # Use the first available domain as fallback
                first_domain = list(self.vector_tool_map.keys())[0]
                selected_domains = [first_domain]
                domain_confidences[first_domain] = 0.5
        
        return {
            "selected_domains": selected_domains,
            "domain_confidences": domain_confidences,
            "reasoning": reasoning_output.reasoning_trace,
        }
    
    def _execute_vector_searches(self, query: str, routing_decision: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute vector searches on the selected knowledge domains.
        
        Args:
            query: The user query
            routing_decision: Routing decision with selected domains
            
        Returns:
            Dict[str, Any]: Search results from selected domains
        """
        selected_domains = routing_decision["selected_domains"]
        domain_confidences = routing_decision["domain_confidences"]
        
        search_results = {}
        tool_calls = []
        
        # Execute search on each selected domain
        for domain in selected_domains:
            if domain in self.vector_tool_map:
                tool = self.vector_tool_map[domain]
                
                # Execute vector search
                try:
                    result = self.execute_tool(domain, query=query)
                    tool_calls.append(result)
                    
                    # Store search results with domain confidence
                    search_results[domain] = {
                        "results": result.get("result", []),
                        "confidence": domain_confidences.get(domain, 0.0),
                        "success": result.get("success", False),
                    }
                    
                except Exception as e:
                    logger.error(f"Error executing vector search on domain '{domain}': {str(e)}")
                    search_results[domain] = {
                        "results": [],
                        "confidence": 0.0,
                        "success": False,
                        "error": str(e),
                    }
        
        return {
            "domain_results": search_results,
            "tool_calls": tool_calls,
        }
    
    def _generate_response(
        self,
        query: str,
        search_results: Dict[str, Any],
        routing_decision: Dict[str, Any],
        context: Dict[str, Any],
    ) -> AgentResponse:
        """
        Generate a response based on the search results.
        
        Args:
            query: The user query
            search_results: Results from vector searches
            routing_decision: Routing decision with selected domains
            context: Additional context
            
        Returns:
            AgentResponse: The agent's response
        """
        # Extract search results from selected domains
        domain_results = search_results["domain_results"]
        tool_calls = search_results["tool_calls"]
        
        # Prepare context for response generation
        context_info = []
        
        # Add search results from each domain as context
        for domain, result in domain_results.items():
            if result["success"] and result["results"]:
                context_info.append(f"Results from {domain}:")
                
                for i, item in enumerate(result["results"][:3]):  # Limit to top 3 results
                    content = item.get("content", "")
                    context_info.append(f"[{i+1}] {content[:200]}...")  # Truncate long content
        
        # Create reasoning input for generating the final response
        reasoning_input = ReasoningInput(
            prompt=query,
            context=context_info,
            system_prompt=(
                "You are a helpful knowledge assistant that provides answers based on the "
                "retrieved information. Ensure your response is informative, accurate, and "
                "directly addresses the user's query based on the search results provided."
            ),
            memory_items=self._get_relevant_memory_items(query),
        )
        
        # Execute reasoning to generate response
        reasoning_output = self.reasoning_engine.reason(reasoning_input)
        
        # Create agent response
        return AgentResponse(
            content=reasoning_output.response,
            success=True,
            reasoning=reasoning_output.reasoning_trace,
            tool_calls=tool_calls,
            metadata={
                "selected_domains": routing_decision["selected_domains"],
                "domain_confidences": routing_decision["domain_confidences"],
            },
        )
    
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