"""
Web search tool implementation for agentic systems.

This tool allows agents to search for information on the web,
enabling access to up-to-date information beyond their training data.
"""

from typing import Any, Dict, List, Optional, Union
import logging
import requests
import json
import os

from .base_tool import BaseTool

logger = logging.getLogger(__name__)

class WebSearchTool(BaseTool):
    """
    Tool for searching the web for information.
    
    This tool enables agents to search for information on the web,
    providing access to up-to-date information and external knowledge.
    """
    
    def __init__(
        self,
        name: str = "web_search",
        description: str = "Search the web for information",
        search_engine: str = "ddg",  # Options: "ddg" (DuckDuckGo), "google", "custom"
        api_key: Optional[str] = None,
        custom_search_url: Optional[str] = None,
        max_results: int = 5,
    ):
        """
        Initialize web search tool.
        
        Args:
            name: Name of the tool
            description: Description of what the tool does
            search_engine: Search engine to use
            api_key: API key for search engine (if required)
            custom_search_url: URL for custom search engine
            max_results: Maximum number of results to return
        """
        parameters = {
            "query": {
                "type": "string",
                "description": "The search query",
            },
            "num_results": {
                "type": "integer",
                "description": "Number of results to return (optional)",
            },
            "site": {
                "type": "string",
                "description": "Limit search to specific site (optional)",
            },
        }
        
        required_parameters = ["query"]
        
        return_schema = {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "title": {
                        "type": "string",
                        "description": "The title of the search result",
                    },
                    "url": {
                        "type": "string",
                        "description": "The URL of the search result",
                    },
                    "snippet": {
                        "type": "string",
                        "description": "A snippet or description of the search result",
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
        
        self.search_engine = search_engine
        self.api_key = api_key or os.environ.get("SEARCH_API_KEY")
        self.custom_search_url = custom_search_url
        self.max_results = max_results
    
    def _execute(self, query: str, num_results: Optional[int] = None, site: Optional[str] = None) -> List[Dict[str, str]]:
        """
        Execute the web search.
        
        Args:
            query: The search query
            num_results: Number of results to return (optional)
            site: Limit search to specific site (optional)
            
        Returns:
            List[Dict[str, str]]: The search results
        """
        # Use provided num_results or default
        effective_num_results = num_results if num_results is not None else self.max_results
        
        # Modify query if site is specified
        if site:
            query = f"site:{site} {query}"
        
        # Select search method based on engine
        if self.search_engine.lower() == "ddg":
            results = self._search_duckduckgo(query, effective_num_results)
        elif self.search_engine.lower() == "google":
            if not self.api_key:
                raise ValueError("API key is required for Google search")
            results = self._search_google(query, effective_num_results)
        elif self.search_engine.lower() == "custom":
            if not self.custom_search_url:
                raise ValueError("Custom search URL is required for custom search")
            results = self._search_custom(query, effective_num_results)
        else:
            raise ValueError(f"Unsupported search engine: {self.search_engine}")
        
        return results
    
    def _search_duckduckgo(self, query: str, num_results: int) -> List[Dict[str, str]]:
        """
        Search using DuckDuckGo.
        
        Args:
            query: The search query
            num_results: Number of results to return
            
        Returns:
            List[Dict[str, str]]: The search results
        """
        try:
            # Use the DuckDuckGo API-like endpoint
            url = "https://api.duckduckgo.com/"
            params = {
                "q": query,
                "format": "json",
                "pretty": 1,
                "no_redirect": 1,
                "no_html": 1,
                "skip_disambig": 1,
            }
            
            response = requests.get(url, params=params)
            response.raise_for_status()
            data = response.json()
            
            results = []
            
            # Extract relevant information from the response
            if "AbstractText" in data and data["AbstractText"]:
                results.append({
                    "title": data.get("Heading", ""),
                    "url": data.get("AbstractURL", ""),
                    "snippet": data["AbstractText"],
                })
            
            # Add results from related topics
            for topic in data.get("RelatedTopics", [])[:num_results - len(results)]:
                if "Text" in topic and "FirstURL" in topic:
                    results.append({
                        "title": topic.get("Text", "").split(" - ")[0],
                        "url": topic.get("FirstURL", ""),
                        "snippet": topic.get("Text", ""),
                    })
                elif "Topics" in topic:
                    for subtopic in topic["Topics"]:
                        if len(results) >= num_results:
                            break
                        if "Text" in subtopic and "FirstURL" in subtopic:
                            results.append({
                                "title": subtopic.get("Text", "").split(" - ")[0],
                                "url": subtopic.get("FirstURL", ""),
                                "snippet": subtopic.get("Text", ""),
                            })
            
            # Limit to requested number of results
            return results[:num_results]
            
        except Exception as e:
            logger.exception(f"Error during DuckDuckGo search: {str(e)}")
            raise
    
    def _search_google(self, query: str, num_results: int) -> List[Dict[str, str]]:
        """
        Search using Google Custom Search API.
        
        Args:
            query: The search query
            num_results: Number of results to return
            
        Returns:
            List[Dict[str, str]]: The search results
        """
        try:
            # Google Custom Search API
            url = "https://www.googleapis.com/customsearch/v1"
            params = {
                "key": self.api_key,
                "cx": os.environ.get("GOOGLE_CSE_ID", ""),  # Custom Search Engine ID
                "q": query,
                "num": min(num_results, 10),  # API limit is 10 per request
            }
            
            response = requests.get(url, params=params)
            response.raise_for_status()
            data = response.json()
            
            results = []
            
            # Extract search results
            for item in data.get("items", [])[:num_results]:
                results.append({
                    "title": item.get("title", ""),
                    "url": item.get("link", ""),
                    "snippet": item.get("snippet", ""),
                })
            
            return results
            
        except Exception as e:
            logger.exception(f"Error during Google search: {str(e)}")
            raise
    
    def _search_custom(self, query: str, num_results: int) -> List[Dict[str, str]]:
        """
        Search using a custom search API.
        
        Args:
            query: The search query
            num_results: Number of results to return
            
        Returns:
            List[Dict[str, str]]: The search results
        """
        try:
            # Custom search implementation
            headers = {}
            if self.api_key:
                headers["Authorization"] = f"Bearer {self.api_key}"
            
            params = {
                "query": query,
                "limit": num_results,
            }
            
            response = requests.post(self.custom_search_url, headers=headers, json=params)
            response.raise_for_status()
            data = response.json()
            
            # Format depends on the custom API
            if isinstance(data, list):
                # If the API returns a list of results directly
                results = []
                for item in data[:num_results]:
                    results.append({
                        "title": item.get("title", ""),
                        "url": item.get("url", ""),
                        "snippet": item.get("snippet", item.get("description", "")),
                    })
                return results
            elif "results" in data and isinstance(data["results"], list):
                # If the API returns a structure with a "results" field
                results = []
                for item in data["results"][:num_results]:
                    results.append({
                        "title": item.get("title", ""),
                        "url": item.get("url", ""),
                        "snippet": item.get("snippet", item.get("description", "")),
                    })
                return results
            else:
                # Fallback for other response formats
                logger.warning(f"Unexpected response format from custom search API: {data}")
                return []
            
        except Exception as e:
            logger.exception(f"Error during custom search: {str(e)}")
            raise 