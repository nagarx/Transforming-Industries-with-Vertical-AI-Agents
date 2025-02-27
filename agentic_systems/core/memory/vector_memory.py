"""
Vector memory implementation for agentic systems.

This memory module uses vector embeddings to store and retrieve information,
enabling semantic search and similarity-based retrieval.
"""

import json
import os
import uuid
from typing import Any, Dict, List, Optional, Union, Tuple

import numpy as np
from sentence_transformers import SentenceTransformer

from .base_memory import BaseMemory, MemoryItem

class VectorMemory(BaseMemory):
    """
    Vector memory implementation using embeddings for semantic search.
    
    This memory module stores items with vector embeddings, enabling
    semantic search and similarity-based retrieval of information.
    """
    
    def __init__(
        self,
        embedding_model: Union[str, SentenceTransformer] = "sentence-transformers/all-MiniLM-L6-v2",
        dimension: Optional[int] = None,
        similarity_threshold: float = 0.7,
    ):
        """
        Initialize vector memory.
        
        Args:
            embedding_model: Model to use for generating embeddings (string path or loaded model)
            dimension: Dimension of the embeddings (optional, determined from model if not provided)
            similarity_threshold: Threshold for similarity search (0.0 to 1.0)
        """
        # Initialize the embedding model
        self.embedding_model = (
            embedding_model if isinstance(embedding_model, SentenceTransformer)
            else SentenceTransformer(embedding_model)
        )
        
        # Determine embedding dimension
        self.dimension = dimension or self.embedding_model.get_sentence_embedding_dimension()
        
        # Set similarity threshold
        self.similarity_threshold = similarity_threshold
        
        # Storage for items and embeddings
        self.items: Dict[str, MemoryItem] = {}
        self.embeddings: Dict[str, np.ndarray] = {}
    
    def _get_embedding(self, text: str) -> np.ndarray:
        """
        Generate embedding for a text.
        
        Args:
            text: The text to embed
            
        Returns:
            np.ndarray: The embedding vector
        """
        return self.embedding_model.encode(text, show_progress_bar=False)
    
    def _similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """
        Calculate cosine similarity between two embeddings.
        
        Args:
            embedding1: First embedding
            embedding2: Second embedding
            
        Returns:
            float: Cosine similarity score
        """
        return np.dot(embedding1, embedding2) / (np.linalg.norm(embedding1) * np.linalg.norm(embedding2))
    
    def add(self, item: Union[MemoryItem, Any], **metadata) -> str:
        """
        Add an item to vector memory.
        
        Args:
            item: The memory item or content to add
            **metadata: Additional metadata to associate with the item
            
        Returns:
            str: The ID of the added memory item
        """
        if not isinstance(item, MemoryItem):
            item = MemoryItem(content=item, metadata=metadata)
        
        if item.memory_id is None:
            item.memory_id = str(uuid.uuid4())
        
        # Store the item
        self.items[item.memory_id] = item
        
        # Generate and store the embedding
        if isinstance(item.content, str):
            self.embeddings[item.memory_id] = self._get_embedding(item.content)
        elif "text" in item.metadata:
            # If content is not a string but metadata contains a text field, use that
            self.embeddings[item.memory_id] = self._get_embedding(item.metadata["text"])
        else:
            # For non-text content, use the string representation if possible
            try:
                self.embeddings[item.memory_id] = self._get_embedding(str(item.content))
            except Exception as e:
                raise ValueError(f"Could not generate embedding for item: {e}")
        
        return item.memory_id
    
    def get(self, memory_id: str) -> Optional[MemoryItem]:
        """
        Retrieve a specific memory item by ID.
        
        Args:
            memory_id: The ID of the memory item to retrieve
            
        Returns:
            Optional[MemoryItem]: The memory item if found, None otherwise
        """
        return self.items.get(memory_id)
    
    def search(self, query: Any, limit: int = 5, **filters) -> List[MemoryItem]:
        """
        Search for memory items using semantic similarity.
        
        Args:
            query: The search query
            limit: Maximum number of results to return
            **filters: Additional filters to apply to metadata
            
        Returns:
            List[MemoryItem]: The search results
        """
        # Handle empty memory
        if not self.items:
            return []
        
        # Convert query to embedding
        query_text = str(query)
        query_embedding = self._get_embedding(query_text)
        
        # Calculate similarities and filter by metadata
        similarities: List[Tuple[str, float]] = []
        
        for memory_id, embedding in self.embeddings.items():
            item = self.items[memory_id]
            
            # Apply metadata filters
            if not all(item.metadata.get(key) == value for key, value in filters.items()):
                continue
            
            # Calculate similarity
            similarity = self._similarity(query_embedding, embedding)
            
            # Add to results if above threshold
            if similarity >= self.similarity_threshold:
                similarities.append((memory_id, similarity))
        
        # Sort by similarity (highest first) and take top-k
        similarities.sort(key=lambda x: x[1], reverse=True)
        top_k = similarities[:limit]
        
        # Return the memory items
        return [self.items[memory_id] for memory_id, _ in top_k]
    
    def update(self, memory_id: str, item: Union[MemoryItem, Any], **metadata) -> bool:
        """
        Update a memory item.
        
        Args:
            memory_id: The ID of the memory item to update
            item: The new memory item or content
            **metadata: Updated metadata
            
        Returns:
            bool: True if the update was successful, False otherwise
        """
        if memory_id not in self.items:
            return False
        
        # Remove old item and embedding
        old_item = self.items[memory_id]
        del self.embeddings[memory_id]
        
        # Add new item with same ID
        if isinstance(item, MemoryItem):
            item.memory_id = memory_id
            self.items[memory_id] = item
        else:
            # Update the content and metadata
            updated_metadata = old_item.metadata.copy()
            if metadata:
                updated_metadata.update(metadata)
            
            self.items[memory_id] = MemoryItem(
                content=item,
                metadata=updated_metadata,
                memory_id=memory_id
            )
        
        # Generate and store the new embedding
        new_item = self.items[memory_id]
        if isinstance(new_item.content, str):
            self.embeddings[memory_id] = self._get_embedding(new_item.content)
        elif "text" in new_item.metadata:
            self.embeddings[memory_id] = self._get_embedding(new_item.metadata["text"])
        else:
            try:
                self.embeddings[memory_id] = self._get_embedding(str(new_item.content))
            except Exception as e:
                # Restore the old item if embedding fails
                self.items[memory_id] = old_item
                self.embeddings[memory_id] = self._get_embedding(str(old_item.content))
                return False
        
        return True
    
    def delete(self, memory_id: str) -> bool:
        """
        Delete a memory item.
        
        Args:
            memory_id: The ID of the memory item to delete
            
        Returns:
            bool: True if the deletion was successful, False otherwise
        """
        if memory_id not in self.items:
            return False
        
        # Remove the item and its embedding
        del self.items[memory_id]
        del self.embeddings[memory_id]
        
        return True
    
    def clear(self) -> None:
        """Clear all memory items."""
        self.items.clear()
        self.embeddings.clear()
    
    def load(self, path: str) -> None:
        """
        Load memory from a directory.
        
        Args:
            path: The directory path to load from
        """
        # Ensure the directory exists
        if not os.path.isdir(path):
            raise ValueError(f"Directory not found: {path}")
        
        # Clear current memory
        self.clear()
        
        # Load items from JSON
        items_path = os.path.join(path, "items.json")
        if not os.path.isfile(items_path):
            raise ValueError(f"Items file not found: {items_path}")
        
        try:
            with open(items_path, 'r') as f:
                items_data = json.load(f)
            
            for item_data in items_data:
                memory_item = MemoryItem.from_dict(item_data)
                self.items[memory_item.memory_id] = memory_item
        
        except (FileNotFoundError, json.JSONDecodeError) as e:
            raise ValueError(f"Failed to load items from {items_path}: {e}")
        
        # Load embeddings from NumPy files
        embeddings_dir = os.path.join(path, "embeddings")
        if not os.path.isdir(embeddings_dir):
            raise ValueError(f"Embeddings directory not found: {embeddings_dir}")
        
        for memory_id in self.items:
            embedding_path = os.path.join(embeddings_dir, f"{memory_id}.npy")
            try:
                self.embeddings[memory_id] = np.load(embedding_path)
            except Exception as e:
                raise ValueError(f"Failed to load embedding for {memory_id}: {e}")
    
    def save(self, path: str) -> None:
        """
        Save memory to a directory.
        
        Args:
            path: The directory path to save to
        """
        # Create directory if it doesn't exist
        os.makedirs(path, exist_ok=True)
        
        # Create embeddings directory
        embeddings_dir = os.path.join(path, "embeddings")
        os.makedirs(embeddings_dir, exist_ok=True)
        
        # Save items to JSON
        items_data = [item.to_dict() for item in self.items.values()]
        items_path = os.path.join(path, "items.json")
        
        try:
            with open(items_path, 'w') as f:
                json.dump(items_data, f, indent=2)
        
        except (PermissionError, IOError) as e:
            raise ValueError(f"Failed to save items to {items_path}: {e}")
        
        # Save embeddings to NumPy files
        for memory_id, embedding in self.embeddings.items():
            embedding_path = os.path.join(embeddings_dir, f"{memory_id}.npy")
            try:
                np.save(embedding_path, embedding)
            except Exception as e:
                raise ValueError(f"Failed to save embedding for {memory_id}: {e}") 