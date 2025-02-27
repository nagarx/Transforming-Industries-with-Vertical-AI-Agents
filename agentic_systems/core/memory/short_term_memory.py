"""
Short-term memory implementation for agentic systems.

This memory module is designed for maintaining recent conversation history
and short-lived contextual information.
"""

import json
import uuid
from typing import Any, Dict, List, Optional, Union
from collections import deque
from datetime import datetime, timedelta

from .base_memory import BaseMemory, MemoryItem

class ShortTermMemory(BaseMemory):
    """
    Short-term memory implementation using a fixed-size queue.
    
    This memory is designed to store recent interactions and context,
    with automatic pruning of older items when capacity is reached.
    """
    
    def __init__(self, max_items: int = 10, expiry_seconds: Optional[int] = None):
        """
        Initialize short-term memory.
        
        Args:
            max_items: Maximum number of items to store
            expiry_seconds: Optional expiry time in seconds for memory items
        """
        self.max_items = max_items
        self.expiry_seconds = expiry_seconds
        self.items: deque = deque(maxlen=max_items)
        self.id_map: Dict[str, MemoryItem] = {}
    
    def add(self, item: Union[MemoryItem, Any], **metadata) -> str:
        """
        Add an item to short-term memory.
        
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
        
        # Remove expired items
        self._prune_expired()
        
        # Add the new item
        self.items.append(item)
        self.id_map[item.memory_id] = item
        
        return item.memory_id
    
    def get(self, memory_id: str) -> Optional[MemoryItem]:
        """
        Retrieve a specific memory item by ID.
        
        Args:
            memory_id: The ID of the memory item to retrieve
            
        Returns:
            Optional[MemoryItem]: The memory item if found, None otherwise
        """
        return self.id_map.get(memory_id)
    
    def search(self, query: Any, limit: int = 5, **filters) -> List[MemoryItem]:
        """
        Search for memory items.
        
        For short-term memory, this is a simple filtered scan of recent items.
        
        Args:
            query: The search query (used for keyword matching in content)
            limit: Maximum number of results to return
            **filters: Additional filters to apply to metadata
            
        Returns:
            List[MemoryItem]: The search results
        """
        self._prune_expired()
        
        results = []
        query_str = str(query).lower()
        
        # Special case to match all items
        if query_str == "*":
            for item in reversed(self.items):  # Most recent first
                # Check if metadata matches filters
                metadata_match = all(item.metadata.get(key) == value for key, value in filters.items() if key != "type")
                
                # If type filter is provided, check content type
                type_match = True
                if "type" in filters and isinstance(item.content, dict):
                    type_match = item.content.get("type") == filters["type"]
                
                if metadata_match and type_match:
                    results.append(item)
                    if len(results) >= limit:
                        break
            return results
                
        # Check if content matches query
        for item in reversed(self.items):  # Most recent first
            match_found = False
            
            # If content is a string, do direct matching
            if isinstance(item.content, str) and query_str in item.content.lower():
                match_found = True
            # If content is a dictionary, search in its values
            elif isinstance(item.content, dict):
                for value in item.content.values():
                    if isinstance(value, str) and query_str in value.lower():
                        match_found = True
                        break
            
            if match_found:
                # Check if metadata matches filters
                metadata_match = all(item.metadata.get(key) == value for key, value in filters.items() if key != "type")
                
                # If type filter is provided, check content type
                type_match = True
                if "type" in filters and isinstance(item.content, dict):
                    type_match = item.content.get("type") == filters["type"]
                
                if metadata_match and type_match:
                    results.append(item)
                    if len(results) >= limit:
                        break
        
        return results
    
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
        existing_item = self.id_map.get(memory_id)
        if not existing_item:
            return False
        
        if isinstance(item, MemoryItem):
            # Preserve the original memory_id
            item.memory_id = memory_id
            self.id_map[memory_id] = item
            
            # Replace in the deque
            for i, old_item in enumerate(self.items):
                if old_item.memory_id == memory_id:
                    self.items[i] = item
                    break
        else:
            # Update the content and metadata
            existing_item.content = item
            if metadata:
                existing_item.metadata.update(metadata)
            # Update the timestamp
            existing_item.timestamp = datetime.now()
        
        return True
    
    def delete(self, memory_id: str) -> bool:
        """
        Delete a memory item.
        
        Args:
            memory_id: The ID of the memory item to delete
            
        Returns:
            bool: True if the deletion was successful, False otherwise
        """
        if memory_id not in self.id_map:
            return False
        
        # Remove from the deque
        item_to_remove = self.id_map[memory_id]
        self.items.remove(item_to_remove)
        
        # Remove from the id map
        del self.id_map[memory_id]
        
        return True
    
    def clear(self) -> None:
        """Clear all memory items."""
        self.items.clear()
        self.id_map.clear()
    
    def get_recent(self, limit: int = None) -> List[MemoryItem]:
        """
        Get the most recent memory items.
        
        Args:
            limit: Maximum number of items to return (defaults to all items)
            
        Returns:
            List[MemoryItem]: The most recent memory items
        """
        self._prune_expired()
        items = list(self.items)
        if limit is not None:
            items = items[-limit:]
        return items
    
    def get_all(self) -> List[MemoryItem]:
        """
        Get all memory items.
        
        Returns:
            List[MemoryItem]: All memory items
        """
        self._prune_expired()
        return list(self.items)
    
    def _prune_expired(self) -> None:
        """Remove expired items from memory."""
        if self.expiry_seconds is None:
            return
        
        cutoff_time = datetime.now() - timedelta(seconds=self.expiry_seconds)
        
        # Identify expired items
        expired_items = [item for item in self.items if item.timestamp < cutoff_time]
        
        # Remove expired items
        for item in expired_items:
            if item.memory_id in self.id_map:
                del self.id_map[item.memory_id]
                self.items.remove(item)
    
    def load(self, path: str) -> None:
        """
        Load memory from a JSON file.
        
        Args:
            path: The path to load from
        """
        try:
            with open(path, 'r') as f:
                data = json.load(f)
            
            self.clear()
            
            for item_data in data:
                memory_item = MemoryItem.from_dict(item_data)
                self.add(memory_item)
        
        except (FileNotFoundError, json.JSONDecodeError) as e:
            raise ValueError(f"Failed to load memory from {path}: {e}")
    
    def save(self, path: str) -> None:
        """
        Save memory to a JSON file.
        
        Args:
            path: The path to save to
        """
        data = [item.to_dict() for item in self.items]
        
        try:
            with open(path, 'w') as f:
                json.dump(data, f, indent=2)
        
        except (PermissionError, IOError) as e:
            raise ValueError(f"Failed to save memory to {path}: {e}") 