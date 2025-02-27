"""
Base tool interface for agentic systems.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union
import time
import logging

logger = logging.getLogger(__name__)

class ToolResponse:
    """Response from a tool execution."""
    
    def __init__(
        self,
        success: bool,
        result: Any,
        error: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize tool response.
        
        Args:
            success: Whether the tool execution was successful
            result: The result of the tool execution
            error: Error message if the execution failed
            metadata: Additional metadata about the execution
        """
        self.success = success
        self.result = result
        self.error = error
        self.metadata = metadata or {}
        self.timestamp = time.time()
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert tool response to dictionary.
        
        Returns:
            Dict[str, Any]: Dictionary representation of the tool response
        """
        return {
            "success": self.success,
            "result": self.result,
            "error": self.error,
            "metadata": self.metadata,
            "timestamp": self.timestamp,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ToolResponse":
        """
        Create tool response from dictionary.
        
        Args:
            data: Dictionary with response data
            
        Returns:
            ToolResponse: Tool response instance
        """
        response = cls(
            success=data["success"],
            result=data["result"],
            error=data.get("error"),
            metadata=data.get("metadata", {}),
        )
        response.timestamp = data.get("timestamp", time.time())
        return response

class BaseTool(ABC):
    """
    Abstract base class for tool implementations.
    
    Tools are external capabilities that agents can use to interact
    with systems, retrieve information, or perform actions.
    """
    
    def __init__(
        self,
        name: str,
        description: str,
        parameters: Optional[Dict[str, Any]] = None,
        required_parameters: Optional[List[str]] = None,
        return_schema: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize base tool.
        
        Args:
            name: Name of the tool
            description: Description of what the tool does
            parameters: Parameters the tool accepts
            required_parameters: List of required parameter names
            return_schema: Schema for the tool's return value
        """
        self.name = name
        self.description = description
        self.parameters = parameters or {}
        self.required_parameters = required_parameters or []
        self.return_schema = return_schema or {}
    
    def execute(self, **kwargs) -> ToolResponse:
        """
        Execute the tool with the given parameters.
        
        Args:
            **kwargs: Parameters for tool execution
            
        Returns:
            ToolResponse: Response from the tool execution
        """
        # Validate required parameters
        missing_params = [param for param in self.required_parameters if param not in kwargs]
        if missing_params:
            error_msg = f"Missing required parameters: {', '.join(missing_params)}"
            logger.error(f"Tool {self.name}: {error_msg}")
            return ToolResponse(success=False, result=None, error=error_msg)
        
        # Execute the tool
        try:
            start_time = time.time()
            result = self._execute(**kwargs)
            execution_time = time.time() - start_time
            
            return ToolResponse(
                success=True,
                result=result,
                metadata={"execution_time": execution_time}
            )
            
        except Exception as e:
            error_msg = f"Error executing tool: {str(e)}"
            logger.exception(f"Tool {self.name} execution failed: {error_msg}")
            return ToolResponse(success=False, result=None, error=error_msg)
    
    @abstractmethod
    def _execute(self, **kwargs) -> Any:
        """
        Actual implementation of the tool execution.
        
        Args:
            **kwargs: Parameters for tool execution
            
        Returns:
            Any: Result of the tool execution
        """
        pass
    
    def get_schema(self) -> Dict[str, Any]:
        """
        Get the schema for this tool.
        
        Returns:
            Dict[str, Any]: Tool schema
        """
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters,
            "required_parameters": self.required_parameters,
            "return_schema": self.return_schema,
        } 