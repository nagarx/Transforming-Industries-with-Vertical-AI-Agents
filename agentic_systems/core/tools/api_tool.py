"""
API tool implementation for agentic systems.

This tool allows agents to interact with external REST APIs,
enabling integration with various services and data sources.
"""

from typing import Any, Dict, List, Optional, Union
import logging
import requests
import json
import os

from .base_tool import BaseTool

logger = logging.getLogger(__name__)

class ApiTool(BaseTool):
    """
    Tool for interacting with external REST APIs.
    
    This tool enables agents to make HTTP requests to external APIs,
    supporting various authentication methods and request types.
    """
    
    def __init__(
        self,
        name: str,
        description: str,
        base_url: str,
        endpoints: Dict[str, Dict[str, Any]],
        auth_type: Optional[str] = None,  # "basic", "bearer", "api_key", "oauth", None
        auth_params: Optional[Dict[str, str]] = None,
        headers: Optional[Dict[str, str]] = None,
        timeout: int = 30,
    ):
        """
        Initialize API tool.
        
        Args:
            name: Name of the tool
            description: Description of what the tool does
            base_url: Base URL for the API
            endpoints: Dictionary of API endpoints with their methods, paths, and parameter schemas
            auth_type: Authentication type to use
            auth_params: Authentication parameters
            headers: Default headers to include in requests
            timeout: Request timeout in seconds
        """
        parameters = {
            "endpoint": {
                "type": "string",
                "description": "The endpoint to call",
                "enum": list(endpoints.keys()),
            },
            "params": {
                "type": "object",
                "description": "Parameters for the API call",
            },
        }
        
        required_parameters = ["endpoint"]
        
        # Build return schema based on endpoints
        return_schema = {
            "type": "object",
            "properties": {
                "status_code": {
                    "type": "integer",
                    "description": "HTTP status code of the response",
                },
                "headers": {
                    "type": "object",
                    "description": "Response headers",
                },
                "data": {
                    "type": "object",
                    "description": "Response data (parsed JSON if applicable)",
                },
                "text": {
                    "type": "string",
                    "description": "Raw response text",
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
        
        self.base_url = base_url.rstrip("/")
        self.endpoints = endpoints
        self.auth_type = auth_type
        self.auth_params = auth_params or {}
        self.headers = headers or {}
        self.timeout = timeout
    
    def _execute(self, endpoint: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Execute the API call.
        
        Args:
            endpoint: The endpoint to call
            params: Parameters for the API call
            
        Returns:
            Dict[str, Any]: The API response
        """
        if endpoint not in self.endpoints:
            raise ValueError(f"Unknown endpoint: {endpoint}")
        
        # Get endpoint configuration
        endpoint_config = self.endpoints[endpoint]
        method = endpoint_config.get("method", "GET").upper()
        path = endpoint_config.get("path", "")
        
        # Build URL
        url = f"{self.base_url}/{path.lstrip('/')}"
        
        # Prepare request
        request_kwargs = self._prepare_request(method, params or {}, endpoint_config)
        
        try:
            # Make the request
            response = requests.request(
                method=method,
                url=url,
                timeout=self.timeout,
                **request_kwargs
            )
            
            # Parse response
            result = self._parse_response(response)
            return result
            
        except Exception as e:
            logger.exception(f"Error during API call to {endpoint}: {str(e)}")
            raise
    
    def _prepare_request(self, method: str, params: Dict[str, Any], endpoint_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Prepare request arguments based on method and parameters.
        
        Args:
            method: HTTP method
            params: Request parameters
            endpoint_config: Endpoint configuration
            
        Returns:
            Dict[str, Any]: Request keyword arguments
        """
        request_kwargs = {}
        
        # Add authentication
        headers = self.headers.copy()
        if self.auth_type:
            self._add_auth_headers(headers)
        
        # Add custom headers from endpoint config
        if "headers" in endpoint_config and isinstance(endpoint_config["headers"], dict):
            headers.update(endpoint_config["headers"])
        
        request_kwargs["headers"] = headers
        
        # Handle parameters based on method
        if method in ["GET", "DELETE"]:
            request_kwargs["params"] = params
        elif method in ["POST", "PUT", "PATCH"]:
            # Check if we should send as JSON or form data
            content_type = headers.get("Content-Type", "").lower()
            if content_type == "application/x-www-form-urlencoded":
                request_kwargs["data"] = params
            else:
                request_kwargs["json"] = params
        
        return request_kwargs
    
    def _add_auth_headers(self, headers: Dict[str, str]) -> None:
        """
        Add authentication headers based on auth type.
        
        Args:
            headers: Headers dictionary to modify
        """
        if self.auth_type == "basic":
            username = self.auth_params.get("username", "")
            password = self.auth_params.get("password", "")
            from base64 import b64encode
            auth_str = f"{username}:{password}"
            encoded = b64encode(auth_str.encode()).decode()
            headers["Authorization"] = f"Basic {encoded}"
        
        elif self.auth_type == "bearer":
            token = self.auth_params.get("token", "")
            headers["Authorization"] = f"Bearer {token}"
        
        elif self.auth_type == "api_key":
            key_name = self.auth_params.get("key_name", "api_key")
            key_value = self.auth_params.get("key_value", "")
            headers[key_name] = key_value
        
        elif self.auth_type == "oauth":
            # For OAuth, we assume the token has been obtained elsewhere
            token = self.auth_params.get("access_token", "")
            headers["Authorization"] = f"Bearer {token}"
    
    def _parse_response(self, response: requests.Response) -> Dict[str, Any]:
        """
        Parse the API response.
        
        Args:
            response: The HTTP response
            
        Returns:
            Dict[str, Any]: Parsed response
        """
        result = {
            "status_code": response.status_code,
            "headers": dict(response.headers),
            "text": response.text,
        }
        
        # Try to parse as JSON
        try:
            result["data"] = response.json()
        except json.JSONDecodeError:
            # If not JSON, include the raw text
            result["data"] = None
        
        return result
    
    def add_endpoint(self, name: str, config: Dict[str, Any]) -> None:
        """
        Add a new endpoint to the API tool.
        
        Args:
            name: Name of the endpoint
            config: Endpoint configuration
        """
        self.endpoints[name] = config
    
    def remove_endpoint(self, name: str) -> bool:
        """
        Remove an endpoint from the API tool.
        
        Args:
            name: Name of the endpoint
            
        Returns:
            bool: True if the endpoint was removed, False otherwise
        """
        if name in self.endpoints:
            del self.endpoints[name]
            return True
        return False
    
    def get_available_endpoints(self) -> List[Dict[str, Any]]:
        """
        Get the list of available endpoints with their descriptions.
        
        Returns:
            List[Dict[str, Any]]: Available endpoints
        """
        return [
            {
                "name": name,
                "method": config.get("method", "GET"),
                "path": config.get("path", ""),
                "description": config.get("description", ""),
                "parameters": config.get("parameters", {}),
            }
            for name, config in self.endpoints.items()
        ] 