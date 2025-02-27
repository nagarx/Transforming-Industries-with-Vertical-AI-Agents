"""
Tests for the memory module.
"""

import unittest
from unittest.mock import MagicMock

from agentic_systems.core.memory import ShortTermMemory, MemoryItem


class TestShortTermMemory(unittest.TestCase):
    """Test cases for the ShortTermMemory class."""

    def setUp(self):
        """Set up test fixtures."""
        self.memory = ShortTermMemory(max_items=5)

    def test_initialization(self):
        """Test memory initialization."""
        self.assertEqual(self.memory.max_items, 5)
        self.assertEqual(len(self.memory.items), 0)

    def test_add_item(self):
        """Test adding an item to memory."""
        item_content = {"type": "message", "content": "Test message", "timestamp": 1234567890}
        memory_id = self.memory.add(item_content)
        
        self.assertEqual(len(self.memory.items), 1)
        self.assertIsInstance(self.memory.items[0], MemoryItem)
        self.assertEqual(self.memory.items[0].content, item_content)
        self.assertIsNotNone(memory_id)

    def test_max_items_limit(self):
        """Test that memory respects the max_items limit."""
        # Add more items than the max_items limit
        memory_ids = []
        for i in range(10):
            content = {"type": "message", "content": f"Test message {i}", "timestamp": 1234567890 + i}
            memory_id = self.memory.add(content)
            memory_ids.append(memory_id)
        
        # Check that only max_items are stored
        self.assertEqual(len(self.memory.items), 5)
        # Check that the oldest items were removed (should have the last 5 messages)
        self.assertEqual(self.memory.items[0].content["content"], "Test message 5")

    def test_get_items(self):
        """Test retrieving items from memory."""
        # Add some items
        memory_ids = []
        for i in range(3):
            content = {"type": "message", "content": f"Test message {i}", "timestamp": 1234567890 + i}
            memory_id = self.memory.add(content)
            memory_ids.append(memory_id)
        
        # Get a specific item by ID
        item = self.memory.get(memory_ids[0])
        self.assertIsNotNone(item)
        self.assertEqual(item.content["content"], "Test message 0")
        
        # Test searching
        query_items = self.memory.search("Test message", limit=10)
        self.assertEqual(len(query_items), 3)
        
        # Add an item with different type
        query_content = {"type": "query", "content": "Test query", "timestamp": 1234567899}
        self.memory.add(query_content)
        
        # Search with type filter
        message_items = self.memory.search("*", limit=10, type="message")
        self.assertEqual(len(message_items), 3)
        
        query_items = self.memory.search("*", limit=10, type="query")
        self.assertEqual(len(query_items), 1)

    def test_clear(self):
        """Test clearing memory."""
        # Add some items
        for i in range(3):
            content = {"type": "message", "content": f"Test message {i}", "timestamp": 1234567890 + i}
            self.memory.add(content)
        
        # Clear memory
        self.memory.clear()
        self.assertEqual(len(self.memory.items), 0)


if __name__ == "__main__":
    unittest.main() 