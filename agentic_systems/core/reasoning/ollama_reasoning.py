"""
Ollama reasoning engine implementation for agentic systems.

This reasoning engine uses the Ollama API to interact with local LLMs
such as the deepseek-r1:14b model.
"""

import json
import re
import requests
from typing import Any, Dict, List, Optional, Union
import logging

from .base_reasoning import BaseReasoning, ReasoningInput, ReasoningOutput

logger = logging.getLogger(__name__)

class OllamaReasoning(BaseReasoning):
    """
    Reasoning engine implementation that uses Ollama API.
    
    This engine connects to locally running Ollama models, providing
    a reasoning capability powered by models like deepseek-r1:14b.
    """
    
    def __init__(
        self,
        model_name: str = "deepseek-r1:14b",
        base_url: str = "http://localhost:11434",
        temperature: float = 0.7,
        max_tokens: int = 2048,
        top_p: float = 0.9,
        top_k: int = 40,
        system_prompt_template: Optional[str] = None,
    ):
        """
        Initialize Ollama reasoning engine.
        
        Args:
            model_name: Name of the Ollama model to use
            base_url: Base URL of the Ollama API
            temperature: Temperature for sampling
            max_tokens: Maximum tokens to generate
            top_p: Top-p sampling parameter
            top_k: Top-k sampling parameter
            system_prompt_template: Template for system prompt
        """
        self.model_name = model_name
        self.base_url = base_url.rstrip("/")
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.top_p = top_p
        self.top_k = top_k
        
        # Default system prompt template if none provided
        self.system_prompt_template = system_prompt_template or (
            "You are an intelligent agent designed to help with various tasks. "
            "You analyze information, make decisions, and provide clear responses. "
            "{context}"
            "{persona}"
        )
    
    def _call_ollama_api(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """
        Call the Ollama API.
        
        Args:
            prompt: The prompt to send to the API
            system_prompt: Optional system prompt
            
        Returns:
            str: The response from the API
            
        Raises:
            Exception: If the API call fails
        """
        url = f"{self.base_url}/api/generate"
        
        # Prepare the payload
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "top_p": self.top_p,
            "top_k": self.top_k,
        }
        
        # Add system prompt if provided
        if system_prompt:
            payload["system"] = system_prompt
        
        try:
            response = requests.post(url, json=payload)
            response.raise_for_status()
            
            # Parse the streaming response
            full_response = ""
            for line in response.text.splitlines():
                if not line.strip():
                    continue
                
                try:
                    data = json.loads(line)
                    if "response" in data:
                        full_response += data["response"]
                except json.JSONDecodeError:
                    logger.warning(f"Failed to parse response line: {line}")
            
            return full_response
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Error calling Ollama API: {e}")
            raise Exception(f"Failed to call Ollama API: {e}")
    
    def reason(self, input_data: ReasoningInput) -> ReasoningOutput:
        """
        Perform reasoning using the Ollama model.
        
        Args:
            input_data: Input data for reasoning
            
        Returns:
            ReasoningOutput: The reasoning output
        """
        # Generate or use provided system prompt
        system_prompt = input_data.system_prompt or self.generate_system_prompt(
            persona=input_data.persona, 
            context="\n".join(input_data.context) if input_data.context else None
        )
        
        # Format the prompt
        formatted_prompt = self.format_prompt(input_data)
        
        # Call the Ollama API
        response_text = self._call_ollama_api(formatted_prompt, system_prompt)
        
        # Parse the response
        parsed_response = self.parse_response(response_text)
        
        # Extract reasoning trace if available
        reasoning_trace = parsed_response.get("reasoning_trace", None)
        response = parsed_response.get("response", response_text)
        next_actions = parsed_response.get("next_actions", [])
        
        # Create and return the reasoning output
        return ReasoningOutput(
            response=response,
            reasoning_trace=reasoning_trace,
            next_actions=next_actions,
            metadata={"model": self.model_name, "raw_response": response_text}
        )
    
    def generate_system_prompt(self, persona: Optional[str] = None, context: Optional[str] = None) -> str:
        """
        Generate a system prompt for the Ollama model.
        
        Args:
            persona: Persona to adopt for reasoning
            context: Additional context to include
            
        Returns:
            str: The generated system prompt
        """
        # Replace placeholders in the template
        return self.system_prompt_template.format(
            persona=f"Adopt the following persona: {persona}\n" if persona else "",
            context=f"Use the following context to inform your reasoning: {context}\n" if context else "",
        )
    
    def parse_response(self, response: str) -> Dict[str, Any]:
        """
        Parse a response from the Ollama model.
        
        This method looks for structured sections in the response:
        - Reasoning: Text between <reasoning> and </reasoning> tags
        - Actions: JSON array between <actions> and </actions> tags
        
        Args:
            response: The raw response from the model
            
        Returns:
            Dict[str, Any]: The parsed response
        """
        result = {"response": response}
        
        # Extract reasoning trace
        reasoning_match = re.search(r"<reasoning>(.*?)</reasoning>", response, re.DOTALL)
        if reasoning_match:
            result["reasoning_trace"] = reasoning_match.group(1).strip()
            # Remove reasoning section from response
            result["response"] = re.sub(r"<reasoning>.*?</reasoning>", "", result["response"], flags=re.DOTALL).strip()
        
        # Extract actions
        actions_match = re.search(r"<actions>(.*?)</actions>", response, re.DOTALL)
        if actions_match:
            actions_text = actions_match.group(1).strip()
            try:
                result["next_actions"] = json.loads(actions_text)
            except json.JSONDecodeError:
                logger.warning(f"Failed to parse actions JSON: {actions_text}")
                result["next_actions"] = [{"type": "text", "content": actions_text}]
            
            # Remove actions section from response
            result["response"] = re.sub(r"<actions>.*?</actions>", "", result["response"], flags=re.DOTALL).strip()
        
        return result
    
    def format_prompt(self, input_data: ReasoningInput) -> str:
        """
        Format a prompt for the Ollama model.
        
        Args:
            input_data: Input data for reasoning
            
        Returns:
            str: The formatted prompt
        """
        prompt_parts = []
        
        # Add memory items if available
        if input_data.memory_items:
            memory_text = "\n\nRelevant memory items:\n"
            for item in input_data.memory_items:
                content = item.get("content", "")
                memory_text += f"- {content}\n"
            prompt_parts.append(memory_text)
        
        # Add tool results if available
        if input_data.tool_results:
            tools_text = "\n\nTool results:\n"
            for result in input_data.tool_results:
                tool_name = result.get("tool_name", "")
                tool_result = result.get("result", "")
                tools_text += f"- {tool_name}: {tool_result}\n"
            prompt_parts.append(tools_text)
        
        # Add cognitive skill results if available
        if input_data.cognitive_skill_results:
            skills_text = "\n\nCognitive skill results:\n"
            for result in input_data.cognitive_skill_results:
                skill_name = result.get("skill_name", "")
                skill_result = result.get("result", "")
                skills_text += f"- {skill_name}: {skill_result}\n"
            prompt_parts.append(skills_text)
        
        # Add context if available
        if input_data.context:
            context_text = "\n\nContext:\n" + "\n".join(input_data.context)
            prompt_parts.append(context_text)
        
        # Add user prompt
        prompt_parts.append(f"\n\nUser query: {input_data.prompt}")
        
        # Add instructions for structured output if needed
        prompt_parts.append(
            "\n\nIf you need to show your reasoning process, wrap it in <reasoning> </reasoning> tags."
            "\nIf you need to specify actions to take, wrap a JSON array in <actions> </actions> tags."
        )
        
        return "".join(prompt_parts) 