# Legal Analysis Multi-Agent System Example

This example demonstrates an orchestrated multi-agent system designed for comprehensive legal analysis. The system coordinates multiple specialized agents, each with expertise in different legal domains, to analyze complex legal cases.

## Key Features

- **Multi-Expert Collaboration**: Combines insights from experts in patent law, contract law, and trade secrets law
- **Orchestrated Analysis**: A lead orchestrator agent coordinates specialized agents and synthesizes their analyses
- **Risk Assessment**: Identifies potential legal risks in analyses
- **Compliance Monitoring**: Ensures legal analyses meet ethical and professional standards
- **Specialized Knowledge Extraction**: Each agent extracts and focuses on relevant information from legal documents

## Running the Example

To run this example, follow these steps:

1. Pull the deepseek-r1:14b model using Ollama:
   ```
   ollama pull deepseek-r1:14b
   ```

2. Make sure Ollama is running on the default port (11434)

3. Run the interactive demo:
   ```
   cd agentic-systems
   python examples/legal_analysis/multi_agent_legal_system.py
   ```

4. Follow the on-screen instructions to analyze the sample legal case

## Example Queries

The system comes with a sample case "Smith v. Technology Corp" that involves patent infringement, trade secret misappropriation, and breach of contract claims. You can analyze this case with queries such as:

- "Analyze the strength of the patent infringement claim"
- "What are the key legal issues in this case?"
- "What defenses might be most effective for Technology Corp?"
- "Assess the potential damages if the plaintiff prevails"
- "What evidence will be critical during discovery?"

The system will coordinate responses from multiple specialized agents to provide a comprehensive legal analysis.

## Implementation Details

This example demonstrates several key components of agentic systems:

1. **RagOrchestratedSystem**: An implementation of a multi-agent system with orchestration
2. **SpecializedLegalAgent**: Custom agent implementation for legal domain expertise
3. **AgentNode**: Structure for organizing agents in a multi-agent system
4. **ChainOfThoughtReasoning**: Enhanced reasoning capabilities for complex analysis tasks
5. **RiskAssessmentSkill & ComplianceMonitoringSkill**: Cognitive skills for ensuring quality analyses

The system demonstrates how multiple specialized agents can collaborate to tackle complex tasks that benefit from diverse expertise and perspectives. Each agent contributes based on their domain knowledge, and the orchestrator synthesizes these contributions into a cohesive analysis. 