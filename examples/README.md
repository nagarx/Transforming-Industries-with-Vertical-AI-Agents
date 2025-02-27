# Agentic Systems Examples

This directory contains example implementations of various agentic systems using our framework. Each example demonstrates different agent architectures and applications to solve specific business problems.

## Available Examples

### [Customer Support](./customer_support/)
A Human-in-the-Loop (HITL) agent implementation for customer support. This example demonstrates how human review and intervention can be integrated into an autonomous agent workflow to handle sensitive customer inquiries.

### [Knowledge Management](./knowledge_management/)
A RAG Router Agent implementation for managing and retrieving information from multiple knowledge domains. This example shows how to route queries to the most relevant knowledge sources based on semantic understanding.

### [Legal Analysis](./legal_analysis/)
A RAG Orchestrated Multi-Agent System for comprehensive legal analysis. This example demonstrates how multiple specialized agents can collaborate to analyze complex legal cases, with each agent contributing domain-specific expertise.

## Using the Examples

Each example is designed to work with the local deepseek-r1:14b model through Ollama. To run the examples:

1. Install Ollama from [https://ollama.com/](https://ollama.com/)

2. Pull the deepseek-r1:14b model:
   ```
   ollama pull deepseek-r1:14b
   ```

3. Navigate to the specific example directory and follow the instructions in its README file.

## Agent Types Demonstrated

These examples showcase the three main types of agents implemented in our framework:

1. **Task-Specific Agents**: Specialized agents focused on one type of task (Knowledge Management example)
2. **Human-Augmented Agents**: Agents that incorporate human feedback into their workflow (Customer Support example)
3. **Multi-Agent Systems**: Coordination of multiple agents working together (Legal Analysis example)

Each example includes a detailed README with implementation details, usage instructions, and example interactions.

## Requirements

- Python 3.8+
- Ollama
- All dependencies listed in the project's main requirements.txt file

For specialized dependencies required by specific examples, check the README in each example directory. 