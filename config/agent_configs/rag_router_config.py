"""
Configuration for the RAG Router agent pattern.
"""

from pydantic import Field
from typing import Dict, Any, List, Optional
from .base_agent_config import BaseAgentConfig, AgentType, AgentPattern, ToolConfig

class VectorDatabaseConfig(BaseAgentConfig):
    """Configuration for a vector database in the RAG Router."""
    database_id: str = Field(..., description="Unique identifier for the vector database")
    database_name: str = Field(..., description="Human-readable name for the vector database")
    domain: str = Field(..., description="Domain of knowledge represented by this database")
    description: str = Field(..., description="Description of the database content and purpose")
    connection_string: str = Field(..., description="Connection string or path to the vector database")
    embedding_model: str = Field(default="sentence-transformers/all-MiniLM-L6-v2", description="Embedding model used for this database")
    search_top_k: int = Field(default=3, description="Number of top results to retrieve")

class RAGRouterConfig(BaseAgentConfig):
    """Configuration for the RAG Router agent pattern."""
    vector_databases: List[VectorDatabaseConfig] = Field(default_factory=list, description="Vector databases available to the router")
    default_database_id: Optional[str] = Field(default=None, description="Default database to use if no clear routing decision can be made")
    confidence_threshold: float = Field(default=0.7, description="Confidence threshold for routing decisions")
    combine_results: bool = Field(default=False, description="Whether to combine results from multiple databases in case of uncertainty")
    
    # Additional router-specific settings
    use_query_reformulation: bool = Field(default=True, description="Whether to reformulate queries before routing")
    max_reformulations: int = Field(default=1, description="Maximum number of query reformulations to attempt")

def create_default_rag_router_config() -> RAGRouterConfig:
    """Create a default configuration for a RAG Router agent."""
    return RAGRouterConfig(
        agent_id="default_rag_router",
        agent_name="Default RAG Router",
        agent_type=AgentType.TASK_SPECIFIC,
        agent_pattern=AgentPattern.RAG_ROUTER,
        description="A router agent that directs queries to the appropriate domain-specific knowledge sources.",
        model_config_id="router",
        vector_databases=[],
        tools=[
            ToolConfig(
                tool_id="query_rewriter",
                tool_type="text_processing",
                description="Rewrites user queries to improve retrieval performance",
            ),
            ToolConfig(
                tool_id="intent_classifier",
                tool_type="classification",
                description="Classifies the intent of user queries to assist with routing decisions",
            )
        ],
        custom_settings={
            "logging_level": "INFO",
            "explain_routing_decision": True,
        }
    )

# Example configuration for a legal domain RAG Router
LEGAL_RAG_ROUTER_CONFIG = RAGRouterConfig(
    agent_id="legal_rag_router",
    agent_name="Legal Knowledge Router",
    agent_type=AgentType.TASK_SPECIFIC,
    agent_pattern=AgentPattern.RAG_ROUTER,
    description="A router agent that directs legal queries to the appropriate legal domain knowledge sources.",
    model_config_id="router",
    vector_databases=[
        VectorDatabaseConfig(
            database_id="contract_law_db",
            database_name="Contract Law Database",
            domain="contract_law",
            description="Knowledge base for contract law, agreements, and related precedents",
            connection_string="db/vector/contract_law",
            embedding_model="sentence-transformers/all-MiniLM-L6-v2",
        ),
        VectorDatabaseConfig(
            database_id="intellectual_property_db",
            database_name="Intellectual Property Database",
            domain="ip_law",
            description="Knowledge base for intellectual property, patents, trademarks, and copyright",
            connection_string="db/vector/ip_law",
            embedding_model="sentence-transformers/all-MiniLM-L6-v2",
        ),
        VectorDatabaseConfig(
            database_id="regulatory_compliance_db",
            database_name="Regulatory Compliance Database",
            domain="regulatory_compliance",
            description="Knowledge base for regulatory compliance across industries",
            connection_string="db/vector/regulatory",
            embedding_model="sentence-transformers/all-MiniLM-L6-v2",
        ),
    ],
    tools=[
        ToolConfig(
            tool_id="legal_query_rewriter",
            tool_type="text_processing",
            description="Rewrites legal queries to improve retrieval performance",
        ),
        ToolConfig(
            tool_id="legal_domain_classifier",
            tool_type="classification",
            description="Classifies the legal domain of user queries to assist with routing decisions",
        )
    ],
    confidence_threshold=0.75,
    default_database_id="regulatory_compliance_db",
    custom_settings={
        "logging_level": "INFO",
        "explain_routing_decision": True,
        "legal_jurisdiction": "US",
        "include_citations": True,
    }
) 