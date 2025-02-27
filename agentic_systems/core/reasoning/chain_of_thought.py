"""
Chain of Thought reasoning implementation for agentic systems.

This reasoning engine extends the Ollama reasoning with explicit step-by-step
reasoning capabilities, enabling more complex problem-solving and decision-making.
"""

from typing import Any, Dict, List, Optional, Union
import logging

from .ollama_reasoning import OllamaReasoning
from .base_reasoning import ReasoningInput, ReasoningOutput

logger = logging.getLogger(__name__)

class ChainOfThoughtReasoning(OllamaReasoning):
    """
    Chain of Thought reasoning engine implementation.
    
    This engine extends the Ollama reasoning engine with step-by-step
    thinking capabilities, enabling more complex reasoning chains.
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
        reasoning_steps: int = 3,
        include_reasoning_in_output: bool = True,
    ):
        """
        Initialize Chain of Thought reasoning engine.
        
        Args:
            model_name: Name of the Ollama model to use
            base_url: Base URL of the Ollama API
            temperature: Temperature for sampling
            max_tokens: Maximum tokens to generate
            top_p: Top-p sampling parameter
            top_k: Top-k sampling parameter
            system_prompt_template: Template for system prompt
            reasoning_steps: Number of reasoning steps to perform
            include_reasoning_in_output: Whether to include reasoning steps in output
        """
        # Default system prompt with CoT instructions if none provided
        if system_prompt_template is None:
            system_prompt_template = (
                "You are an intelligent agent designed to solve complex problems using a chain of thought approach. "
                "When faced with a question or task, break down your reasoning into clear, logical steps. "
                "Think step-by-step and explain your reasoning process explicitly. "
                "{context}"
                "{persona}"
            )
        
        super().__init__(
            model_name=model_name,
            base_url=base_url,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            top_k=top_k,
            system_prompt_template=system_prompt_template,
        )
        
        self.reasoning_steps = reasoning_steps
        self.include_reasoning_in_output = include_reasoning_in_output
    
    def reason(self, input_data: ReasoningInput) -> ReasoningOutput:
        """
        Perform chain of thought reasoning.
        
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
        
        # Format the prompt with CoT instructions
        formatted_prompt = self._format_cot_prompt(input_data)
        
        # Initialize reasoning steps
        reasoning_steps = []
        current_context = formatted_prompt
        
        # Perform reasoning steps
        for step in range(self.reasoning_steps):
            # Call the Ollama API for this step
            step_prompt = (
                current_context + 
                f"\n\nStep {step + 1}/{self.reasoning_steps} of reasoning:" if step > 0 
                else current_context
            )
            
            response_text = self._call_ollama_api(step_prompt, system_prompt)
            
            # Parse the response
            parsed_response = self.parse_response(response_text)
            
            # Extract reasoning trace if available
            step_reasoning = parsed_response.get("reasoning_trace", response_text)
            reasoning_steps.append(step_reasoning)
            
            # Check if we've reached a conclusion
            if parsed_response.get("next_actions") or "conclusion" in response_text.lower():
                break
            
            # Add this step's reasoning to the context for the next step
            current_context = step_prompt + f"\n\nIntermediate reasoning: {step_reasoning}"
        
        # Extract final response and actions from the last step
        final_parsed_response = self.parse_response(reasoning_steps[-1])
        response = final_parsed_response.get("response", reasoning_steps[-1])
        next_actions = final_parsed_response.get("next_actions", [])
        
        # Combine all reasoning steps
        full_reasoning = "\n\n".join([f"Step {i+1}: {step}" for i, step in enumerate(reasoning_steps)])
        
        # Create the output
        return ReasoningOutput(
            response=response,
            reasoning_trace=full_reasoning if self.include_reasoning_in_output else None,
            next_actions=next_actions,
            metadata={
                "model": self.model_name,
                "reasoning_steps": reasoning_steps,
                "step_count": len(reasoning_steps),
            }
        )
    
    def _format_cot_prompt(self, input_data: ReasoningInput) -> str:
        """
        Format a Chain of Thought prompt.
        
        Args:
            input_data: Input data for reasoning
            
        Returns:
            str: The formatted CoT prompt
        """
        # Start with the base formatted prompt
        prompt = super().format_prompt(input_data)
        
        # Add Chain of Thought specific instructions
        cot_instructions = (
            "\n\nTo solve this problem, please follow these steps:"
            "\n1. Analyze and understand the problem or question."
            "\n2. Break down the problem into smaller parts if necessary."
            "\n3. Work through each part systematically, showing your reasoning."
            "\n4. Consider different approaches or perspectives."
            "\n5. Evaluate your solutions and check for errors."
            "\n6. Provide a clear final answer or conclusion."
            "\n\nPlease think step by step and be explicit about your reasoning process."
        )
        
        return prompt + cot_instructions
    
    def generate_system_prompt(self, persona: Optional[str] = None, context: Optional[str] = None) -> str:
        """
        Generate a system prompt with Chain of Thought instructions.
        
        Args:
            persona: Persona to adopt for reasoning
            context: Additional context to include
            
        Returns:
            str: The generated system prompt
        """
        # Generate the base system prompt
        system_prompt = super().generate_system_prompt(persona, context)
        
        # Add specific CoT instructions if not already included
        if "chain of thought" not in system_prompt.lower() and "step by step" not in system_prompt.lower():
            system_prompt += (
                "\n\nWhen solving problems, use a chain of thought approach. "
                "Break down your reasoning into explicit steps and explain your thought process clearly."
            )
        
        return system_prompt 