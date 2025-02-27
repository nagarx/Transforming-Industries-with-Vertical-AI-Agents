"""
Long-term memory implementation for agentic systems.

This memory module extends vector memory to provide persistent storage
with additional features for managing long-term knowledge.
"""

import os
import json
import time
from typing import Any, Dict, List, Optional, Union, Set

from .vector_memory import VectorMemory
from .base_memory import MemoryItem

class LongTermMemory(VectorMemory):
    """
    Long-term memory implementation for persistent knowledge storage.
    
    This memory module extends vector memory with additional features
    for importance weighting, categorization, and periodic reinforcement.
    """
    
    def __init__(
        self,
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        dimension: Optional[int] = None,
        similarity_threshold: float = 0.7,
        storage_path: Optional[str] = None,
        auto_save: bool = True,
        max_items: Optional[int] = None,
    ):
        """
        Initialize long-term memory.
        
        Args:
            embedding_model: Model to use for generating embeddings
            dimension: Dimension of the embeddings (optional)
            similarity_threshold: Threshold for similarity search
            storage_path: Path for persistent storage (optional)
            auto_save: Whether to automatically save after modifications
            max_items: Maximum number of items to store (None for unlimited)
        """
        super().__init__(embedding_model, dimension, similarity_threshold)
        
        self.storage_path = storage_path
        self.auto_save = auto_save
        self.max_items = max_items
        
        # Additional attributes for long-term memory
        self.importance_scores: Dict[str, float] = {}
        self.categories: Dict[str, Set[str]] = {}  # Item ID -> categories
        self.category_items: Dict[str, Set[str]] = {}  # Category -> item IDs
        self.last_accessed: Dict[str, float] = {}  # Item ID -> timestamp
        
        # Load existing memory if path is provided
        if storage_path and os.path.exists(storage_path):
            self.load(storage_path)
    
    def add(self, item: Union[MemoryItem, Any], importance: float = 1.0, categories: List[str] = None, **metadata) -> str:
        """
        Add an item to long-term memory.
        
        Args:
            item: The memory item or content to add
            importance: Importance score for the item (1.0 = normal, higher = more important)
            categories: List of categories to associate with the item
            **metadata: Additional metadata to associate with the item
            
        Returns:
            str: The ID of the added memory item
        """
        # Prepare categories
        categories = categories or []
        
        # Add or update metadata with categories
        if not isinstance(item, MemoryItem):
            metadata["categories"] = categories
        else:
            item.metadata["categories"] = categories
        
        # Add to vector memory
        memory_id = super().add(item, **metadata)
        
        # Store importance score
        self.importance_scores[memory_id] = importance
        
        # Store access timestamp
        self.last_accessed[memory_id] = time.time()
        
        # Store categories
        self.categories[memory_id] = set(categories)
        for category in categories:
            if category not in self.category_items:
                self.category_items[category] = set()
            self.category_items[category].add(memory_id)
        
        # Check if we need to prune
        if self.max_items is not None and len(self.items) > self.max_items:
            self._prune_least_important()
        
        # Autosave if enabled
        if self.auto_save and self.storage_path:
            self.save(self.storage_path)
        
        return memory_id
    
    def get(self, memory_id: str) -> Optional[MemoryItem]:
        """
        Retrieve a specific memory item by ID.
        
        Args:
            memory_id: The ID of the memory item to retrieve
            
        Returns:
            Optional[MemoryItem]: The memory item if found, None otherwise
        """
        item = super().get(memory_id)
        
        if item:
            # Update access timestamp
            self.last_accessed[memory_id] = time.time()
        
        return item
    
    def search(self, query: Any, limit: int = 5, categories: List[str] = None, **filters) -> List[MemoryItem]:
        """
        Search for memory items using semantic similarity with category filtering.
        
        Args:
            query: The search query
            limit: Maximum number of results to return
            categories: Optional list of categories to filter by
            **filters: Additional filters to apply to metadata
            
        Returns:
            List[MemoryItem]: The search results
        """
        # If categories are provided, find items in those categories
        if categories:
            category_ids = set()
            for category in categories:
                if category in self.category_items:
                    category_ids.update(self.category_items[category])
            
            # If no items match the categories, return empty list
            if not category_ids:
                return []
            
            # Filter by category IDs using metadata
            filters["memory_id"] = lambda x: x in category_ids
        
        # Perform search
        results = super().search(query, limit, **filters)
        
        # Update access timestamps for returned items
        for item in results:
            self.last_accessed[item.memory_id] = time.time()
        
        return results
    
    def update(self, memory_id: str, item: Union[MemoryItem, Any], 
              importance: Optional[float] = None, 
              categories: Optional[List[str]] = None, 
              **metadata) -> bool:
        """
        Update a memory item.
        
        Args:
            memory_id: The ID of the memory item to update
            item: The new memory item or content
            importance: New importance score (optional)
            categories: New categories (optional)
            **metadata: Updated metadata
            
        Returns:
            bool: True if the update was successful, False otherwise
        """
        # Update categories in metadata if provided
        if categories is not None:
            if not isinstance(item, MemoryItem):
                metadata["categories"] = categories
            else:
                item.metadata["categories"] = categories
        
        # Update in vector memory
        result = super().update(memory_id, item, **metadata)
        
        if result:
            # Update importance if provided
            if importance is not None:
                self.importance_scores[memory_id] = importance
            
            # Update categories if provided
            if categories is not None:
                # Remove from old categories
                old_categories = self.categories.get(memory_id, set())
                for category in old_categories:
                    if memory_id in self.category_items.get(category, set()):
                        self.category_items[category].remove(memory_id)
                
                # Add to new categories
                self.categories[memory_id] = set(categories)
                for category in categories:
                    if category not in self.category_items:
                        self.category_items[category] = set()
                    self.category_items[category].add(memory_id)
            
            # Update access timestamp
            self.last_accessed[memory_id] = time.time()
            
            # Autosave if enabled
            if self.auto_save and self.storage_path:
                self.save(self.storage_path)
        
        return result
    
    def delete(self, memory_id: str) -> bool:
        """
        Delete a memory item.
        
        Args:
            memory_id: The ID of the memory item to delete
            
        Returns:
            bool: True if the deletion was successful, False otherwise
        """
        # Remove from categories
        if memory_id in self.categories:
            for category in self.categories[memory_id]:
                if category in self.category_items and memory_id in self.category_items[category]:
                    self.category_items[category].remove(memory_id)
            del self.categories[memory_id]
        
        # Remove importance score
        if memory_id in self.importance_scores:
            del self.importance_scores[memory_id]
        
        # Remove access timestamp
        if memory_id in self.last_accessed:
            del self.last_accessed[memory_id]
        
        # Delete from vector memory
        result = super().delete(memory_id)
        
        # Autosave if enabled
        if result and self.auto_save and self.storage_path:
            self.save(self.storage_path)
        
        return result
    
    def clear(self) -> None:
        """Clear all memory items."""
        super().clear()
        
        # Clear additional attributes
        self.importance_scores.clear()
        self.categories.clear()
        self.category_items.clear()
        self.last_accessed.clear()
        
        # Autosave if enabled
        if self.auto_save and self.storage_path:
            self.save(self.storage_path)
    
    def get_by_category(self, category: str, limit: Optional[int] = None) -> List[MemoryItem]:
        """
        Get memory items by category.
        
        Args:
            category: Category to filter by
            limit: Maximum number of items to return (None for all)
            
        Returns:
            List[MemoryItem]: The memory items in the category
        """
        if category not in self.category_items:
            return []
        
        memory_ids = list(self.category_items[category])
        
        # Sort by importance (highest first)
        memory_ids.sort(key=lambda x: self.importance_scores.get(x, 0), reverse=True)
        
        # Apply limit
        if limit is not None:
            memory_ids = memory_ids[:limit]
        
        # Get items
        items = []
        for memory_id in memory_ids:
            item = self.get(memory_id)
            if item:
                items.append(item)
        
        return items
    
    def get_categories(self) -> List[str]:
        """
        Get all categories.
        
        Returns:
            List[str]: All categories
        """
        return list(self.category_items.keys())
    
    def _prune_least_important(self) -> None:
        """Prune the least important items when capacity is reached."""
        if not self.items or self.max_items is None:
            return
        
        # Calculate number of items to remove
        num_to_remove = len(self.items) - self.max_items
        if num_to_remove <= 0:
            return
        
        # Sort items by importance and last access time
        # Items with lower importance and older access times are removed first
        def item_priority(memory_id):
            importance = self.importance_scores.get(memory_id, 0)
            last_access = self.last_accessed.get(memory_id, 0)
            return (importance, last_access)
        
        memory_ids = list(self.items.keys())
        memory_ids.sort(key=item_priority)
        
        # Remove items
        for memory_id in memory_ids[:num_to_remove]:
            self.delete(memory_id)
    
    def save(self, path: str) -> None:
        """
        Save long-term memory to a directory.
        
        Args:
            path: The directory path to save to
        """
        # Save vector memory
        super().save(path)
        
        # Create metadata directory
        metadata_dir = os.path.join(path, "metadata")
        os.makedirs(metadata_dir, exist_ok=True)
        
        # Save importance scores
        importance_path = os.path.join(metadata_dir, "importance.json")
        with open(importance_path, 'w') as f:
            json.dump(self.importance_scores, f, indent=2)
        
        # Save categories (convert sets to lists for JSON serialization)
        categories_path = os.path.join(metadata_dir, "categories.json")
        categories_dict = {k: list(v) for k, v in self.categories.items()}
        with open(categories_path, 'w') as f:
            json.dump(categories_dict, f, indent=2)
        
        # Save category items
        category_items_path = os.path.join(metadata_dir, "category_items.json")
        category_items_dict = {k: list(v) for k, v in self.category_items.items()}
        with open(category_items_path, 'w') as f:
            json.dump(category_items_dict, f, indent=2)
        
        # Save last accessed
        last_accessed_path = os.path.join(metadata_dir, "last_accessed.json")
        with open(last_accessed_path, 'w') as f:
            json.dump(self.last_accessed, f, indent=2)
    
    def load(self, path: str) -> None:
        """
        Load long-term memory from a directory.
        
        Args:
            path: The directory path to load from
        """
        # Load vector memory
        super().load(path)
        
        # Load metadata
        metadata_dir = os.path.join(path, "metadata")
        
        # Load importance scores
        importance_path = os.path.join(metadata_dir, "importance.json")
        if os.path.exists(importance_path):
            with open(importance_path, 'r') as f:
                self.importance_scores = json.load(f)
        
        # Load categories
        categories_path = os.path.join(metadata_dir, "categories.json")
        if os.path.exists(categories_path):
            with open(categories_path, 'r') as f:
                categories_dict = json.load(f)
                self.categories = {k: set(v) for k, v in categories_dict.items()}
        
        # Load category items
        category_items_path = os.path.join(metadata_dir, "category_items.json")
        if os.path.exists(category_items_path):
            with open(category_items_path, 'r') as f:
                category_items_dict = json.load(f)
                self.category_items = {k: set(v) for k, v in category_items_dict.items()}
        
        # Load last accessed
        last_accessed_path = os.path.join(metadata_dir, "last_accessed.json")
        if os.path.exists(last_accessed_path):
            with open(last_accessed_path, 'r') as f:
                self.last_accessed = json.load(f) 