"""
RAG Router Agent for Knowledge Management Example

This example demonstrates a RAG Router agent that routes queries to the appropriate
knowledge domain, using the deepseek-r1:14b local model.
"""

import os
import sys
import logging
from typing import Dict, Any, List, Tuple
import json
import numpy as np

# Add the parent directory to the path to import agentic_systems
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from agentic_systems.agents import RagRouterAgent
from agentic_systems.core.reasoning import OllamaReasoning
from agentic_systems.core.memory import VectorMemory
from agentic_systems.core.tools import VectorSearchTool

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Define knowledge domains
DOMAINS = {
    "hr_policies": {
        "name": "HR Policies",
        "description": "Human resources policies, procedures, and guidelines",
        "documents": [
            {
                "title": "Employee Onboarding Process",
                "content": "The onboarding process includes completing tax forms, setting up email and system accounts, assigning a mentor, and scheduling orientation sessions. New employees should complete all onboarding within the first week."
            },
            {
                "title": "Leave Policy",
                "content": "Employees receive 15 days of paid time off annually, accrued monthly. Unused PTO can roll over up to 5 days to the next year. Sick leave is separate and provides 10 days annually."
            },
            {
                "title": "Remote Work Guidelines",
                "content": "Employees may work remotely up to 3 days per week with manager approval. Remote workers must be available during core hours (10am-3pm) and maintain productivity metrics consistent with in-office work."
            }
        ]
    },
    "technical_documentation": {
        "name": "Technical Documentation",
        "description": "System architecture, API references, and technical guides",
        "documents": [
            {
                "title": "API Authentication",
                "content": "All API requests require OAuth 2.0 authentication. Obtain an access token from /auth/token endpoint using your client ID and secret. Include the token in the Authorization header for all subsequent requests."
            },
            {
                "title": "Database Schema",
                "content": "The system uses a PostgreSQL database with three main schemas: accounts, products, and analytics. The accounts schema contains user and organization tables with one-to-many relationships."
            },
            {
                "title": "Deployment Process",
                "content": "Deployments use a CI/CD pipeline with GitHub Actions. Code merged to the main branch triggers automated testing and staging deployment. Production deployments require manual approval and are scheduled during off-peak hours."
            }
        ]
    },
    "customer_faqs": {
        "name": "Customer FAQs",
        "description": "Frequently asked questions and answers for customers",
        "documents": [
            {
                "title": "Account Management",
                "content": "You can update your account details in the Profile section. To change your password, go to Settings > Security. If you've forgotten your password, use the 'Forgot Password' link on the login screen."
            },
            {
                "title": "Subscription and Billing",
                "content": "We offer monthly and annual billing cycles. Annual subscriptions include a 20% discount. You can upgrade your plan at any time, and the difference will be prorated. Downgrading takes effect at the end of your current billing period."
            },
            {
                "title": "Data Export",
                "content": "To export your data, go to Settings > Data and select 'Export All Data'. The system will generate a ZIP file containing all your data in JSON format, which will be emailed to you when ready."
            }
        ]
    }
}

class InMemoryVectorDB:
    """Simple in-memory vector database for demonstration purposes."""
    
    def __init__(self):
        """Initialize the in-memory vector database."""
        self.vectors = {}  # domain -> list of (vector, metadata) tuples
    
    def add_vectors(self, domain: str, vectors: List[Tuple[np.ndarray, Dict]]):
        """Add vectors to the database."""
        if domain not in self.vectors:
            self.vectors[domain] = []
        self.vectors[domain].extend(vectors)
    
    def search(self, query_vector: np.ndarray, top_k: int = 5, collection: str = None) -> List[Dict]:
        """
        Search for similar vectors.
        
        Args:
            query_vector: Query vector
            top_k: Number of results to return
            collection: Optional domain to search in
            
        Returns:
            List[Dict]: Search results
        """
        results = []
        
        # Determine which domains to search
        domains_to_search = [collection] if collection else list(self.vectors.keys())
        
        # Search in each domain
        for domain in domains_to_search:
            if domain not in self.vectors:
                continue
            
            domain_vectors = self.vectors[domain]
            
            # Calculate cosine similarity for each vector
            similarities = []
            for vector, metadata in domain_vectors:
                similarity = np.dot(query_vector, vector) / (np.linalg.norm(query_vector) * np.linalg.norm(vector))
                similarities.append((similarity, metadata))
            
            # Sort by similarity (highest first)
            similarities.sort(key=lambda x: x[0], reverse=True)
            
            # Add top results
            for similarity, metadata in similarities[:top_k]:
                results.append({
                    "content": metadata["content"],
                    "metadata": metadata,
                    "similarity_score": float(similarity),
                })
        
        # Sort all results by similarity score
        results.sort(key=lambda x: x["similarity_score"], reverse=True)
        
        # Return top_k results
        return results[:top_k]

def setup_vector_db(embedding_model) -> InMemoryVectorDB:
    """
    Set up the vector database with sample documents.
    
    Args:
        embedding_model: Model to generate embeddings
        
    Returns:
        InMemoryVectorDB: Initialized vector database
    """
    vector_db = InMemoryVectorDB()
    
    # Add documents to each domain
    for domain, domain_info in DOMAINS.items():
        vectors = []
        for doc in domain_info["documents"]:
            # Generate embedding for the document
            content = f"{doc['title']}. {doc['content']}"
            embedding = embedding_model.encode(content)
            
            # Add to vectors
            vectors.append((
                embedding,
                {
                    "title": doc["title"],
                    "content": content,
                    "domain": domain,
                }
            ))
        
        # Add vectors to the database
        vector_db.add_vectors(domain, vectors)
    
    return vector_db

def create_knowledge_router():
    """
    Create a RAG Router agent for knowledge management.
    
    Returns:
        RagRouterAgent: Initialized RAG Router agent
    """
    # Initialize the reasoning engine with the local Ollama model
    reasoning_engine = OllamaReasoning(
        model_name="deepseek-r1:14b",
        base_url="http://localhost:11434",
        temperature=0.3,  # Lower temperature for more deterministic routing
        system_prompt_template=(
            "You are a knowledge routing agent that directs queries to the appropriate knowledge domain. "
            "Analyze each query carefully to determine which knowledge domain is most relevant. "
            "Be precise in your routing decisions to ensure users get the most accurate information. "
            "{context}"
            "{persona}"
        )
    )
    
    # Create a vector memory for embeddings
    vector_memory = VectorMemory(
        embedding_model="sentence-transformers/all-MiniLM-L6-v2"
    )
    
    # Set up the vector database
    vector_db = setup_vector_db(vector_memory.embedding_model)
    
    # Create vector search tools for each domain
    vector_search_tools = []
    for domain, domain_info in DOMAINS.items():
        tool = VectorSearchTool(
            name=f"{domain}_search",
            description=f"Search for information in the {domain_info['name']} domain: {domain_info['description']}",
            embedding_model=vector_memory.embedding_model,
            top_k=3,
            similarity_threshold=0.5,
        )
        
        # Set the vector database
        tool.vector_db = vector_db
        
        vector_search_tools.append(tool)
    
    # Create the RAG Router agent
    router = RagRouterAgent(
        agent_id="knowledge_router",
        name="Knowledge Management Router",
        description="Routes queries to the appropriate knowledge domain",
        reasoning_engine=reasoning_engine,
        vector_search_tools=vector_search_tools,
        memory=vector_memory,
        confidence_threshold=0.6,
        combine_results=True,  # Combine results from multiple domains if confidence is low
    )
    
    return router

def interactive_demo():
    """Run an interactive demo of the knowledge router agent."""
    print("\n===== Knowledge Management Router Demo =====")
    print("This demo simulates a knowledge management system with the RAG Router.")
    print("The agent will route your query to the appropriate knowledge domain:\n")
    
    for domain, info in DOMAINS.items():
        print(f"- {info['name']}: {info['description']}")
    
    # Initialize the router
    print("\nInitializing the RAG Router agent...")
    router = create_knowledge_router()
    print("Initialization complete. Ready to answer queries.\n")
    
    # Interactive query loop
    print("Type your knowledge queries below. Type 'exit' to end the demo.\n")
    while True:
        query = input("\nQuery: ")
        if query.lower() in ["exit", "quit", "bye"]:
            print("Ending demo. Goodbye!")
            break
        
        # Execute the router
        logger.info(f"Processing query: {query}")
        response = router.execute(query)
        
        # Display routing information
        selected_domains = response.metadata.get("routing_decision", {}).get("selected_domains", [])
        domain_confidences = response.metadata.get("routing_decision", {}).get("domain_confidences", {})
        
        print("\n----- Routing Information -----")
        print(f"Selected domains: {', '.join(selected_domains)}")
        for domain, confidence in domain_confidences.items():
            if domain in selected_domains:
                print(f"- {domain}: {confidence:.2f} confidence")
        
        print("\n----- Response -----")
        print(response.content)

if __name__ == "__main__":
    interactive_demo() 