"""
Tests for the BaseAgent class.
"""

import unittest
from unittest.mock import MagicMock, patch

from agentic_systems.agents.base_agent import BaseAgent, AgentResponse
from agentic_systems.core.reasoning import ReasoningOutput

# Create a concrete implementation of BaseAgent for testing
class TestAgent(BaseAgent):
    """Concrete implementation of BaseAgent for testing purposes."""
    
    def execute(self, query: str, context=None):
        """Test implementation of the execute method."""
        reasoning_output = self.reasoning_engine.reason(query=query, context=context)
        
        response = AgentResponse(
            content=reasoning_output.response,
            success=True,
            reasoning=reasoning_output.reasoning_trace,
        )
        
        return response


class TestBaseAgent(unittest.TestCase):
    """Test cases for the BaseAgent class."""

    def setUp(self):
        """Set up test fixtures."""
        self.mock_reasoning = MagicMock()
        self.mock_reasoning.reason.return_value = ReasoningOutput(
            response="Test response",
            reasoning_trace="Test reasoning",
            confidence=0.9
        )
        
        # Use the concrete TestAgent implementation instead of abstract BaseAgent
        self.agent = TestAgent(
            agent_id="test_agent",
            name="Test Agent",
            description="A test agent",
            reasoning_engine=self.mock_reasoning
        )

    def test_initialization(self):
        """Test agent initialization."""
        self.assertEqual(self.agent.agent_id, "test_agent")
        self.assertEqual(self.agent.name, "Test Agent")
        self.assertEqual(self.agent.description, "A test agent")
        self.assertEqual(self.agent.reasoning_engine, self.mock_reasoning)
        self.assertEqual(len(self.agent.tools), 0)
        self.assertEqual(len(self.agent.cognitive_skills), 0)
        self.assertIsNotNone(self.agent.memory)  # Now uses ShortTermMemory by default

    def test_execute_basic(self):
        """Test basic execution without tools or skills."""
        response = self.agent.execute("Test query")
        
        # Check that reasoning engine was called
        self.mock_reasoning.reason.assert_called_once()
        
        # Check response
        self.assertIsInstance(response, AgentResponse)
        self.assertEqual(response.content, "Test response")
        self.assertTrue(response.success)
        self.assertEqual(response.reasoning, "Test reasoning")

    def test_add_tool(self):
        """Test adding a tool to the agent."""
        mock_tool = MagicMock()
        mock_tool.name = "test_tool"
        
        self.agent.add_tool(mock_tool)
        self.assertEqual(len(self.agent.tools), 1)
        self.assertEqual(self.agent.tools[0], mock_tool)
        self.assertEqual(self.agent.tool_map["test_tool"], mock_tool)

    def test_add_cognitive_skill(self):
        """Test adding a cognitive skill to the agent."""
        mock_skill = MagicMock()
        mock_skill.name = "test_skill"
        
        self.agent.add_cognitive_skill(mock_skill)
        self.assertEqual(len(self.agent.cognitive_skills), 1)
        self.assertEqual(self.agent.cognitive_skills[0], mock_skill)
        self.assertEqual(self.agent.skill_map["test_skill"], mock_skill)


if __name__ == "__main__":
    unittest.main() 