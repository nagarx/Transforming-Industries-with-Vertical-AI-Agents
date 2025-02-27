"""
Multi-Agent Legal Analysis System Example

This example demonstrates a RAG Orchestrated Multi-Agent system for legal analysis,
using the deepseek-r1:14b local model. The system orchestrates multiple specialized
agents to analyze legal documents and provide comprehensive legal insights.
"""

import os
import sys
import logging
from typing import Dict, Any, List

# Add the parent directory to the path to import agentic_systems
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from agentic_systems.agents import RagOrchestratedSystem, AgentNode, BaseAgent
from agentic_systems.core.reasoning import OllamaReasoning, ChainOfThoughtReasoning
from agentic_systems.core.memory import ShortTermMemory
from agentic_systems.core.cognitive_skills import RiskAssessmentSkill, ComplianceMonitoringSkill

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Sample legal case data
LEGAL_CASE = {
    "title": "Smith v. Technology Corp",
    "case_number": "CV-2023-1234",
    "court": "U.S. District Court, Northern District of California",
    "date_filed": "2023-03-15",
    "plaintiff": "John Smith",
    "defendant": "Technology Corp",
    "allegations": [
        "Patent infringement under 35 U.S.C. § 271",
        "Trade secret misappropriation under 18 U.S.C. § 1836",
        "Breach of contract"
    ],
    "documents": [
        {
            "title": "Complaint",
            "content": "Plaintiff John Smith alleges that Technology Corp infringed on Patent No. 9,876,543 for 'Method and System for Secure Data Transmission' by implementing similar encryption methods in their latest product. Additionally, plaintiff alleges that defendant misappropriated trade secrets after a failed acquisition discussion in 2022, during which confidential information was shared under NDA. The plaintiff seeks damages of $5 million and injunctive relief."
        },
        {
            "title": "Answer",
            "content": "Defendant Technology Corp denies all allegations of patent infringement, asserting that their encryption methods were independently developed and rely on different technical approaches than those described in Patent No. 9,876,543. Defendant also denies misappropriating any trade secrets, stating that all product development was based on publicly available information and their own prior research. Defendant seeks dismissal of all claims and attorney's fees."
        },
        {
            "title": "Patent Description",
            "content": "Patent No. 9,876,543 describes a method for secure data transmission using a multi-layered encryption approach with dynamic key generation. The innovation relies on a specific implementation of elliptic curve cryptography combined with a proprietary key rotation mechanism that operates on a millisecond timescale. The patent was granted in 2020 after a three-year review process."
        }
    ],
    "relevant_laws": [
        "35 U.S.C. § 271 - Patent infringement",
        "18 U.S.C. § 1836 - Trade secret misappropriation",
        "California Civil Code § 1549-1701 - Contract law"
    ],
    "procedural_status": "Discovery phase"
}

class SpecializedLegalAgent(BaseAgent):
    """Base class for specialized legal agents."""
    
    def __init__(self, agent_id, name, description, reasoning_engine, expertise):
        """Initialize specialized legal agent."""
        super().__init__(
            agent_id=agent_id,
            name=name,
            description=description,
            reasoning_engine=reasoning_engine,
            memory=ShortTermMemory(),
        )
        self.expertise = expertise
    
    def execute(self, query, context=None):
        """Execute the agent with the given query and context."""
        context = context or {}
        
        # Add expertise to the agent's reasoning process
        if "case_data" in context:
            case_data = context["case_data"]
            
            # Extract relevant information based on expertise
            if self.expertise == "patent_law":
                relevant_info = {
                    "patent_details": case_data.get("documents", [])[2].get("content", ""),
                    "infringement_allegation": case_data.get("documents", [])[0].get("content", ""),
                    "defense": case_data.get("documents", [])[1].get("content", ""),
                    "relevant_law": case_data.get("relevant_laws", [])[0],
                }
                context["relevant_information"] = relevant_info
                
            elif self.expertise == "contract_law":
                relevant_info = {
                    "contract_allegation": "Breach of contract" in case_data.get("allegations", []),
                    "nda_mention": "NDA" in case_data.get("documents", [])[0].get("content", ""),
                    "relevant_law": case_data.get("relevant_laws", [])[2],
                }
                context["relevant_information"] = relevant_info
                
            elif self.expertise == "trade_secret":
                relevant_info = {
                    "trade_secret_allegation": case_data.get("documents", [])[0].get("content", ""),
                    "defense": case_data.get("documents", [])[1].get("content", ""),
                    "relevant_law": case_data.get("relevant_laws", [])[1],
                }
                context["relevant_information"] = relevant_info
        
        # Use standard reasoning process to generate response
        from agentic_systems.core.reasoning import ReasoningInput
        
        reasoning_input = ReasoningInput(
            prompt=query,
            context=[f"{key}: {value}" for key, value in context.items() if key != "case_data"],
            system_prompt=(
                f"You are a specialized legal expert in {self.expertise}. "
                f"Analyze the given information and provide expert insights based on your specialization. "
                f"Be thorough, precise, and cite relevant laws and precedents where applicable."
            ),
        )
        
        reasoning_output = self.reasoning_engine.reason(reasoning_input)
        
        # Create and return response
        from agentic_systems.agents import AgentResponse
        
        return AgentResponse(
            content=reasoning_output.response,
            success=True,
            reasoning=reasoning_output.reasoning_trace,
            metadata={
                "expertise": self.expertise,
                "confidence": reasoning_output.confidence,
            },
        )

def create_legal_analysis_system():
    """
    Create a multi-agent legal analysis system.
    
    Returns:
        RagOrchestratedSystem: A legal analysis multi-agent system
    """
    # Create specialized reasoning engines
    patent_reasoning = ChainOfThoughtReasoning(
        model_name="deepseek-r1:14b",
        base_url="http://localhost:11434",
        temperature=0.2,  # Low temperature for fact-based reasoning
        reasoning_steps=3,  # More steps for detailed analysis
    )
    
    contract_reasoning = OllamaReasoning(
        model_name="deepseek-r1:14b",
        base_url="http://localhost:11434",
        temperature=0.3,
    )
    
    trade_secret_reasoning = OllamaReasoning(
        model_name="deepseek-r1:14b",
        base_url="http://localhost:11434",
        temperature=0.3,
    )
    
    orchestrator_reasoning = OllamaReasoning(
        model_name="deepseek-r1:14b",
        base_url="http://localhost:11434",
        temperature=0.4,
        system_prompt_template=(
            "You are the lead orchestrator for a legal analysis system. "
            "Your role is to coordinate specialized legal agents, each with different expertise, "
            "to produce comprehensive legal analyses. Ensure that the final analysis is coherent, "
            "balanced, and addresses all relevant legal aspects of the case. "
            "{context}"
            "{persona}"
        )
    )
    
    # Create specialized agents
    patent_agent = SpecializedLegalAgent(
        agent_id="patent_agent",
        name="Patent Law Expert",
        description="Specializes in patent law, including infringement analysis, patentability, and IP protection strategies",
        reasoning_engine=patent_reasoning,
        expertise="patent_law",
    )
    
    contract_agent = SpecializedLegalAgent(
        agent_id="contract_agent",
        name="Contract Law Expert",
        description="Specializes in contract law, including formation, breach, remedies, and interpretation",
        reasoning_engine=contract_reasoning,
        expertise="contract_law",
    )
    
    trade_secret_agent = SpecializedLegalAgent(
        agent_id="trade_secret_agent",
        name="Trade Secret Expert",
        description="Specializes in trade secret law, including misappropriation, protection strategies, and remedies",
        reasoning_engine=trade_secret_reasoning,
        expertise="trade_secret",
    )
    
    # Create agent nodes
    agent_nodes = [
        AgentNode(
            agent=patent_agent,
            role="Patent Analysis",
            description="Analyzes patent infringement claims and defenses",
            is_lead=False,
        ),
        AgentNode(
            agent=contract_agent,
            role="Contract Analysis",
            description="Analyzes breach of contract claims and defenses",
            is_lead=False,
        ),
        AgentNode(
            agent=trade_secret_agent,
            role="Trade Secret Analysis",
            description="Analyzes trade secret misappropriation claims and defenses",
            is_lead=False,
        ),
    ]
    
    # Create orchestrator agent node (lead agent)
    orchestrator_node = AgentNode(
        agent=BaseAgent(
            agent_id="legal_orchestrator",
            name="Legal Analysis Orchestrator",
            description="Coordinates specialized legal agents and synthesizes their analyses",
            reasoning_engine=orchestrator_reasoning,
        ),
        role="Lead Orchestrator",
        description="Coordinates the work of specialized legal agents and synthesizes their analyses",
        is_lead=True,
    )
    
    # Add orchestrator to agent nodes
    agent_nodes.append(orchestrator_node)
    
    # Create risk assessment skill
    risk_assessment = RiskAssessmentSkill(
        name="legal_risk_assessment",
        description="Assesses risk in legal analyses",
        use_llm=True,
        llm_config={
            "model": "deepseek-r1:14b",
            "base_url": "http://localhost:11434",
        }
    )
    
    # Create compliance monitoring skill
    compliance_monitoring = ComplianceMonitoringSkill(
        name="legal_compliance_monitoring",
        description="Monitors compliance with legal standards and ethics",
        use_llm=True,
        llm_config={
            "model": "deepseek-r1:14b",
            "base_url": "http://localhost:11434",
        }
    )
    
    # Create the multi-agent system
    system = RagOrchestratedSystem(
        system_id="legal_analysis_system",
        name="Legal Analysis System",
        description="A multi-agent system for comprehensive legal analysis",
        agents=agent_nodes,
        orchestration_engine=orchestrator_reasoning,
        system_cognitive_skills=[risk_assessment, compliance_monitoring],
        orchestration_strategy="sequential",  # Sequential analysis building on previous insights
        risk_assessment=risk_assessment,
    )
    
    return system

def analyze_case(case_data, query):
    """
    Analyze a legal case with the multi-agent system.
    
    Args:
        case_data: Legal case data
        query: Query about the case
        
    Returns:
        str: Legal analysis
    """
    # Create legal analysis system
    system = create_legal_analysis_system()
    
    # Create context with case data
    context = {
        "case_data": case_data,
        "domain": "legal_analysis",
    }
    
    # Execute the multi-agent system
    logger.info(f"Analyzing case: {case_data['title']}")
    response = system.execute(query, context)
    
    # Return the analysis
    return response.content, response.metadata

def interactive_demo():
    """Run an interactive demo of the legal analysis system."""
    print("\n===== Legal Analysis Multi-Agent System Demo =====")
    print("This demo simulates a legal analysis system with multiple specialized agents.")
    print("The system will analyze a sample legal case: Smith v. Technology Corp\n")
    
    print("Case Overview:")
    print(f"Title: {LEGAL_CASE['title']}")
    print(f"Allegations: {', '.join(LEGAL_CASE['allegations'])}")
    print(f"Procedural Status: {LEGAL_CASE['procedural_status']}\n")
    
    # Display sample queries
    print("Example queries:")
    print("1. Analyze the strength of the patent infringement claim")
    print("2. What are the key legal issues in this case?")
    print("3. What defenses might be most effective for Technology Corp?")
    print("4. Assess the potential damages if the plaintiff prevails")
    print("5. What evidence will be critical during discovery?\n")
    
    # Interactive query loop
    print("Type your legal analysis questions below. Type 'exit' to end the demo.\n")
    while True:
        query = input("\nLegal Query: ")
        if query.lower() in ["exit", "quit", "bye"]:
            print("Ending demo. Goodbye!")
            break
        
        # Analyze the case
        print("\nAnalyzing... (this may take a moment)")
        analysis, metadata = analyze_case(LEGAL_CASE, query)
        
        # Display the analysis
        print("\n----- Legal Analysis -----")
        print(analysis)
        
        # Display which agents contributed (from metadata)
        if "agent_responses" in metadata:
            print("\n----- Contributing Experts -----")
            for agent_id, response in metadata["agent_responses"].items():
                print(f"- {agent_id}")

if __name__ == "__main__":
    interactive_demo() 