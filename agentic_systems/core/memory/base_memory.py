"""
Base memory interface for agentic systems.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union
from datetime import datetime

class MemoryItem:
    """A single item in memory."""
    
    def __init__(
        self, 
        content: Any,
        metadata: Optional[Dict[str, Any]] = None,
        timestamp: Optional[datetime] = None,
        memory_id: Optional[str] = None
    ):
        self.content = content
        self.metadata = metadata or {}
        self.timestamp = timestamp or datetime.now()
        self.memory_id = memory_id
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert memory item to dictionary."""
        return {
            "content": self.content,
            "metadata": self.metadata,
            "timestamp": self.timestamp.isoformat(),
            "memory_id": self.memory_id
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MemoryItem":
        """Create memory item from dictionary."""
        return cls(
            content=data["content"],
            metadata=data["metadata"],
            timestamp=datetime.fromisoformat(data["timestamp"]),
            memory_id=data.get("memory_id")
        )

class BaseMemory(ABC):
    """Abstract base class for memory implementations."""
    
    @abstractmethod
    def add(self, item: Union[MemoryItem, Any], **metadata) -> str:
        """
        Add an item to memory.
        
        Args:
            item: The memory item or content to add
            **metadata: Additional metadata to associate with the item
            
        Returns:
            str: The ID of the added memory item
        """
        pass
    
    @abstractmethod
    def get(self, memory_id: str) -> Optional[MemoryItem]:
        """
        Retrieve a specific memory item by ID.
        
        Args:
            memory_id: The ID of the memory item to retrieve
            
        Returns:
            Optional[MemoryItem]: The memory item if found, None otherwise
        """
        pass
    
    @abstractmethod
    def search(self, query: Any, limit: int = 5, **filters) -> List[MemoryItem]:
        """
        Search for memory items.
        
        Args:
            query: The search query
            limit: Maximum number of results to return
            **filters: Additional filters to apply
            
        Returns:
            List[MemoryItem]: The search results
        """
        pass
    
    @abstractmethod
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
        pass
    
    @abstractmethod
    def delete(self, memory_id: str) -> bool:
        """
        Delete a memory item.
        
        Args:
            memory_id: The ID of the memory item to delete
            
        Returns:
            bool: True if the deletion was successful, False otherwise
        """
        pass
    
    @abstractmethod
    def clear(self) -> None:
        """Clear all memory items."""
        pass
    
    @abstractmethod
    def load(self, path: str) -> None:
        """
        Load memory from a file.
        
        Args:
            path: The path to load from
        """
        pass
    
    @abstractmethod
    def save(self, path: str) -> None:
        """
        Save memory to a file.
        
        Args:
            path: The path to save to
        """
        pass 