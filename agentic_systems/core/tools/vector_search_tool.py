"""
Vector search tool implementation for agentic systems.

This tool allows agents to search for information in vector databases,
enabling retrieval augmented generation (RAG) capabilities.
"""

from typing import Any, Dict, List, Optional, Union
import logging

from sentence_transformers import SentenceTransformer
import numpy as np

from .base_tool import BaseTool

logger = logging.getLogger(__name__)

class VectorSearchTool(BaseTool):
    """
    Tool for searching vector databases.
    
    This tool enables agents to search for information in vector databases
    using semantic queries, supporting retrieval augmented generation.
    """
    
    def __init__(
        self,
        name: str = "vector_search",
        description: str = "Search for information in a vector database using semantic search",
        vector_db: Any = None,
        embedding_model: Union[str, SentenceTransformer] = "sentence-transformers/all-MiniLM-L6-v2",
        top_k: int = 5,
        similarity_threshold: float = 0.6,
    ):
        """
        Initialize vector search tool.
        
        Args:
            name: Name of the tool
            description: Description of what the tool does
            vector_db: Vector database to search
            embedding_model: Model to use for generating query embeddings
            top_k: Number of top results to return
            similarity_threshold: Minimum similarity score for results
        """
        parameters = {
            "query": {
                "type": "string",
                "description": "The search query",
            },
            "collection": {
                "type": "string",
                "description": "The collection to search in (optional)",
            },
            "top_k": {
                "type": "integer",
                "description": "Number of top results to return (optional)",
            },
            "filter": {
                "type": "object",
                "description": "Filter to apply to search results (optional)",
            },
        }
        
        required_parameters = ["query"]
        
        return_schema = {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "content": {
                        "type": "string",
                        "description": "The content of the document",
                    },
                    "metadata": {
                        "type": "object",
                        "description": "Metadata associated with the document",
                    },
                    "similarity_score": {
                        "type": "number",
                        "description": "Similarity score between the query and the document",
                    },
                },
            },
        }
        
        super().__init__(
            name=name,
            description=description,
            parameters=parameters,
            required_parameters=required_parameters,
            return_schema=return_schema,
        )
        
        self.vector_db = vector_db
        self.top_k = top_k
        self.similarity_threshold = similarity_threshold
        
        # Initialize the embedding model
        if isinstance(embedding_model, str):
            logger.info(f"Loading embedding model: {embedding_model}")
            self.embedding_model = SentenceTransformer(embedding_model)
        else:
            self.embedding_model = embedding_model
    
    def _execute(self, query: str, collection: Optional[str] = None, top_k: Optional[int] = None, filter: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        Execute the vector search.
        
        Args:
            query: The search query
            collection: The collection to search in (optional)
            top_k: Number of top results to return (optional)
            filter: Filter to apply to search results (optional)
            
        Returns:
            List[Dict[str, Any]]: The search results
        """
        if self.vector_db is None:
            raise ValueError("Vector database not initialized")
        
        # Use provided top_k or default
        effective_top_k = top_k if top_k is not None else self.top_k
        
        # Generate query embedding
        query_embedding = self.embedding_model.encode(query)
        
        # Search the vector database
        # This is a generic implementation that needs to be adapted based on the specific vector DB
        try:
            if hasattr(self.vector_db, "search") and callable(self.vector_db.search):
                # For vector DBs with a search method (e.g., Chroma, Pinecone)
                search_params = {"top_k": effective_top_k}
                if collection:
                    search_params["collection_name"] = collection
                if filter:
                    search_params["filter"] = filter
                
                results = self.vector_db.search(
                    query_embedding, 
                    **search_params
                )
            elif hasattr(self.vector_db, "similarity_search_with_score") and callable(self.vector_db.similarity_search_with_score):
                # For LangChain vector stores
                search_params = {"k": effective_top_k}
                if filter:
                    search_params["filter"] = filter
                
                results = self.vector_db.similarity_search_with_score(
                    query, 
                    **search_params
                )
            else:
                # Fallback for custom vector DBs
                results = self._custom_search(query_embedding, collection, effective_top_k, filter)
            
            # Format results
            formatted_results = self._format_results(results, query_embedding)
            
            # Filter by similarity threshold
            formatted_results = [
                result for result in formatted_results 
                if result.get("similarity_score", 0) >= self.similarity_threshold
            ]
            
            return formatted_results
            
        except Exception as e:
            logger.exception(f"Error during vector search: {str(e)}")
            raise
    
    def _custom_search(self, query_embedding: np.ndarray, collection: Optional[str] = None, top_k: int = 5, filter: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        Custom search implementation for vector DBs without standard interfaces.
        
        This is a placeholder that should be overridden for specific vector DB implementations.
        
        Args:
            query_embedding: The query embedding
            collection: The collection to search in (optional)
            top_k: Number of top results to return
            filter: Filter to apply to search results (optional)
            
        Returns:
            List[Dict[str, Any]]: The search results
        """
        # This is a placeholder implementation that should be overridden
        raise NotImplementedError("Custom search not implemented for this vector database")
    
    def _format_results(self, results: List[Any], query_embedding: Optional[np.ndarray] = None) -> List[Dict[str, Any]]:
        """
        Format search results into a standard format.
        
        Args:
            results: Raw search results from the vector DB
            query_embedding: The query embedding (for similarity calculation)
            
        Returns:
            List[Dict[str, Any]]: Formatted search results
        """
        formatted_results = []
        
        # This is a generic implementation that needs to be adapted based on the specific vector DB
        for result in results:
            if isinstance(result, tuple) and len(result) == 2:
                # Format for LangChain similarity_search_with_score results
                doc, score = result
                formatted_result = {
                    "content": doc.page_content if hasattr(doc, "page_content") else str(doc),
                    "metadata": doc.metadata if hasattr(doc, "metadata") else {},
                    "similarity_score": float(score),
                }
            elif isinstance(result, dict):
                # Format for typical vector DB results
                formatted_result = {
                    "content": result.get("text", result.get("content", str(result))),
                    "metadata": result.get("metadata", {}),
                    "similarity_score": result.get("score", result.get("similarity", 0.0)),
                }
            else:
                # Fallback for other result formats
                formatted_result = {
                    "content": str(result),
                    "metadata": {},
                    "similarity_score": 0.0,
                }
            
            formatted_results.append(formatted_result)
        
        return formatted_results
    
    def set_vector_db(self, vector_db: Any) -> None:
        """
        Set the vector database to search.
        
        Args:
            vector_db: Vector database to search
        """
        self.vector_db = vector_db 