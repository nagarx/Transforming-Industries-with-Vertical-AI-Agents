# Agentic Systems

A comprehensive implementation of the paper "Agentic Systems: A Guide to Transforming Industries with Vertical AI Agents" by Fouad Bousetouane.

## Overview

This repository provides a technical implementation of agentic systems as described in the paper. It showcases the architecture and components of AI agents powered by Large Language Models (LLMs), particularly using the Ollama deepseek-r1:14b model for local deployment.

## Architecture

The implementation follows the modular architecture described in the paper, consisting of:

### Core Components
- **Memory Module**: Stores context, history, and domain knowledge
- **Reasoning Engine**: LLM-powered decision-making core (using deepseek-r1:14b)
- **Cognitive Skills**: Domain-specific inference capabilities
- **Tools**: External integrations and API connectors

### Agent Types
- **Task-Specific Agents**: Specialized for discrete tasks (e.g., RAG Router)
- **Multi-Agent Systems**: Collaborative frameworks for complex tasks
- **Human-Augmented Agents**: Systems with human oversight and feedback

## Getting Started

### Prerequisites
- Python 3.8+
- [Ollama](https://ollama.ai/) with deepseek-r1:14b installed

### Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/agentic-systems.git
cd agentic-systems
```

2. Install the required dependencies:
```bash
pip install -r requirements.txt
```

3. Install Ollama and pull the deepseek-r1:14b model:
```bash
# Install Ollama from https://ollama.ai/
# Then pull the model:
ollama pull deepseek-r1:14b
```

4. Install the package in development mode:
```bash
pip install -e .
```

### Running Examples

The repository includes three example implementations:

1. **Customer Support** (Human-in-the-Loop Agent):
```bash
python examples/customer_support/hitl_customer_agent.py
```

2. **Knowledge Management** (RAG Router Agent):
```bash
python examples/knowledge_management/rag_router_demo.py
```

3. **Legal Analysis** (Multi-Agent System):
```bash
python examples/legal_analysis/multi_agent_legal_system.py
```

See the [examples directory](./examples) for more details on each implementation.

## Project Structure

```
agentic_systems/
├── agents/                # Agent implementations
│   ├── task_specific/     # Task-specific agents
│   ├── multi_agent/       # Multi-agent systems
│   └── human_augmented/   # Human-augmented agents
├── core/                  # Core components
│   ├── memory/            # Memory implementations
│   ├── reasoning/         # Reasoning engines
│   ├── cognitive_skills/  # Specialized inference capabilities
│   └── tools/             # External integrations
└── examples/              # Example implementations
    ├── customer_support/  # HITL agent for customer support
    ├── knowledge_management/ # RAG Router for knowledge retrieval
    └── legal_analysis/    # Multi-agent system for legal analysis
```

## Documentation

Detailed documentation for each component is available in their respective directories.

## License

MIT

## Citation

If you use this implementation in your work, please cite the original paper:

```
Bousetouane, F. (2024). Agentic Systems: A Guide to Transforming Industries with Vertical AI Agents.
``` 