# Knowledge Management Router Example

This example demonstrates a RAG Router agent for knowledge management using the deepseek-r1:14b local model. The agent routes queries to the appropriate knowledge domain and retrieves relevant information, implementing the RAG (Retrieval Augmented Generation) pattern.

## Key Features

- **Domain-Based Routing**: Routes queries to the most relevant knowledge domain based on semantic understanding.
- **Vector Search**: Uses vector embeddings to find the most relevant documents for a query.
- **Multi-Domain Knowledge**: Handles knowledge across HR policies, technical documentation, and customer FAQs.
- **Confidence-Based Routing**: Can combine results from multiple domains when confidence is low.
- **In-Memory Vector Database**: Includes a simple implementation of a vector database for demonstration purposes.

## Running the Example

1. Make sure you have the deepseek-r1:14b model running in Ollama:
   ```bash
   ollama pull deepseek-r1:14b
   ```

2. Install required dependencies (sentence-transformers for embeddings):
   ```bash
   pip install sentence-transformers
   ```

3. Run the interactive demo:
   ```bash
   cd agentic-systems
   python examples/knowledge_management/rag_router_demo.py
   ```

4. Ask questions about different knowledge domains and observe how the agent routes and answers them.

## Example Queries

Try these example queries to see the router in action:

- "What is the PTO policy?"
- "How does the API authentication work?"
- "Can I export my data from the system?"
- "Tell me about the employee onboarding process."
- "How do I change my password?"
- "What database does the system use?"

Notice how the agent routes each query to the relevant domain (HR Policies, Technical Documentation, or Customer FAQs) and retrieves the most relevant information.

## Implementation Details

This example showcases:

1. **RagRouterAgent**: The implementation of a task-specific agent for knowledge routing.
2. **VectorSearchTool**: Tools for searching different knowledge domains.
3. **VectorMemory**: Implementation of vector-based memory for semantic search.
4. **InMemoryVectorDB**: A simple vector database implementation for demonstration purposes.
5. **Document Embedding**: Converting documents into vector embeddings for semantic search. 