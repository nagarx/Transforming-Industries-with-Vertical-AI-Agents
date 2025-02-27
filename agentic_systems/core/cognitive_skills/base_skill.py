"""
Base cognitive skill interface for agentic systems.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union
import time
import logging

logger = logging.getLogger(__name__)

class SkillResponse:
    """Response from a cognitive skill."""
    
    def __init__(
        self,
        success: bool,
        result: Any,
        confidence: Optional[float] = None,
        error: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize skill response.
        
        Args:
            success: Whether the skill execution was successful
            result: The result of the skill execution
            confidence: Confidence score for the result
            error: Error message if the execution failed
            metadata: Additional metadata about the execution
        """
        self.success = success
        self.result = result
        self.confidence = confidence
        self.error = error
        self.metadata = metadata or {}
        self.timestamp = time.time()
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert skill response to dictionary.
        
        Returns:
            Dict[str, Any]: Dictionary representation of the skill response
        """
        return {
            "success": self.success,
            "result": self.result,
            "confidence": self.confidence,
            "error": self.error,
            "metadata": self.metadata,
            "timestamp": self.timestamp,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SkillResponse":
        """
        Create skill response from dictionary.
        
        Args:
            data: Dictionary with response data
            
        Returns:
            SkillResponse: Skill response instance
        """
        response = cls(
            success=data["success"],
            result=data["result"],
            confidence=data.get("confidence"),
            error=data.get("error"),
            metadata=data.get("metadata", {}),
        )
        response.timestamp = data.get("timestamp", time.time())
        return response

class BaseCognitiveSkill(ABC):
    """
    Abstract base class for cognitive skill implementations.
    
    Cognitive skills are specialized models or algorithms that provide
    domain-specific inference capabilities, enhancing the agent's ability
    to handle complex, domain-specific tasks.
    """
    
    def __init__(
        self,
        name: str,
        description: str,
        version: str = "1.0.0",
        input_schema: Optional[Dict[str, Any]] = None,
        output_schema: Optional[Dict[str, Any]] = None,
        model_path: Optional[str] = None,
    ):
        """
        Initialize base cognitive skill.
        
        Args:
            name: Name of the skill
            description: Description of what the skill does
            version: Version of the skill
            input_schema: Schema for skill inputs
            output_schema: Schema for skill outputs
            model_path: Path to the model file (if applicable)
        """
        self.name = name
        self.description = description
        self.version = version
        self.input_schema = input_schema or {}
        self.output_schema = output_schema or {}
        self.model_path = model_path
        self.loaded = False
    
    @abstractmethod
    def load(self) -> bool:
        """
        Load the cognitive skill model.
        
        Returns:
            bool: True if loading was successful, False otherwise
        """
        pass
    
    def execute(self, input_data: Any) -> SkillResponse:
        """
        Execute the cognitive skill with the given input data.
        
        Args:
            input_data: Input data for the skill
            
        Returns:
            SkillResponse: Response from the skill execution
        """
        # Load model if not already loaded
        if not self.loaded:
            success = self.load()
            if not success:
                return SkillResponse(
                    success=False,
                    result=None,
                    error="Failed to load the cognitive skill model"
                )
        
        # Validate input data against schema (if applicable)
        if self.input_schema and not self._validate_input(input_data):
            return SkillResponse(
                success=False,
                result=None,
                error="Input data does not match the expected schema"
            )
        
        # Execute the skill
        try:
            start_time = time.time()
            result, confidence = self._execute(input_data)
            execution_time = time.time() - start_time
            
            return SkillResponse(
                success=True,
                result=result,
                confidence=confidence,
                metadata={"execution_time": execution_time}
            )
            
        except Exception as e:
            error_msg = f"Error executing cognitive skill: {str(e)}"
            logger.exception(f"Skill {self.name} execution failed: {error_msg}")
            return SkillResponse(success=False, result=None, error=error_msg)
    
    @abstractmethod
    def _execute(self, input_data: Any) -> tuple[Any, Optional[float]]:
        """
        Actual implementation of the skill execution.
        
        Args:
            input_data: Input data for the skill
            
        Returns:
            tuple[Any, Optional[float]]: Result and confidence score
        """
        pass
    
    def _validate_input(self, input_data: Any) -> bool:
        """
        Validate input data against the schema.
        
        This is a simple implementation that should be overridden by
        skills with complex input validation requirements.
        
        Args:
            input_data: Input data to validate
            
        Returns:
            bool: True if the input is valid, False otherwise
        """
        # If no schema is defined, assume the input is valid
        if not self.input_schema:
            return True
        
        # Simple type validation based on schema
        try:
            if self.input_schema.get("type") == "object" and isinstance(input_data, dict):
                required_fields = self.input_schema.get("required", [])
                properties = self.input_schema.get("properties", {})
                
                # Check if all required fields are present
                for field in required_fields:
                    if field not in input_data:
                        logger.warning(f"Missing required field: {field}")
                        return False
                
                # Check field types if properties are defined
                for field, value in input_data.items():
                    if field in properties:
                        field_schema = properties[field]
                        field_type = field_schema.get("type")
                        
                        if field_type == "string" and not isinstance(value, str):
                            logger.warning(f"Field {field} should be a string")
                            return False
                        elif field_type == "integer" and not isinstance(value, int):
                            logger.warning(f"Field {field} should be an integer")
                            return False
                        elif field_type == "number" and not isinstance(value, (int, float)):
                            logger.warning(f"Field {field} should be a number")
                            return False
                        elif field_type == "boolean" and not isinstance(value, bool):
                            logger.warning(f"Field {field} should be a boolean")
                            return False
                        elif field_type == "array" and not isinstance(value, list):
                            logger.warning(f"Field {field} should be an array")
                            return False
                        elif field_type == "object" and not isinstance(value, dict):
                            logger.warning(f"Field {field} should be an object")
                            return False
                
                return True
                
            elif self.input_schema.get("type") == "array" and isinstance(input_data, list):
                return True
                
            elif self.input_schema.get("type") == "string" and isinstance(input_data, str):
                return True
                
            elif self.input_schema.get("type") == "number" and isinstance(input_data, (int, float)):
                return True
                
            elif self.input_schema.get("type") == "boolean" and isinstance(input_data, bool):
                return True
                
            else:
                logger.warning(f"Input data type does not match schema: {self.input_schema.get('type')}")
                return False
                
        except Exception as e:
            logger.exception(f"Error validating input data: {str(e)}")
            return False
        
        return True
    
    def get_schema(self) -> Dict[str, Any]:
        """
        Get the schema for this cognitive skill.
        
        Returns:
            Dict[str, Any]: Skill schema
        """
        return {
            "name": self.name,
            "description": self.description,
            "version": self.version,
            "input_schema": self.input_schema,
            "output_schema": self.output_schema,
        } 