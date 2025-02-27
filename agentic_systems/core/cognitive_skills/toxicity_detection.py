"""
Toxicity detection cognitive skill for agentic systems.

This skill identifies and scores various types of toxicity in text content,
helping agents to avoid generating or processing harmful content.
"""

import os
import json
import logging
import re
from typing import Any, Dict, List, Optional, Tuple, Union
import time

from ..cognitive_skills.base_skill import BaseCognitiveSkill, SkillResponse

logger = logging.getLogger(__name__)

class ToxicityDetectionSkill(BaseCognitiveSkill):
    """
    Cognitive skill for detecting toxic and harmful content in text.
    
    This skill evaluates text for different categories of toxicity, such as
    hate speech, harassment, profanity, threatening language, self-harm,
    sexual content, violence, etc.
    """
    
    TOXICITY_CATEGORIES = [
        "hate_speech",       # Hate speech based on identity attributes
        "harassment",        # Harassment or bullying
        "profanity",         # Profane or vulgar language
        "threatening",       # Threatening language
        "self_harm",         # Content related to self-harm
        "sexual",            # Sexual or explicit content
        "violence",          # Violent content
        "child_unsafe",      # Content unsafe for children
        "misinformation",    # False or misleading information
    ]
    
    def __init__(
        self,
        name: str = "toxicity_detection",
        description: str = "Detects toxic and harmful content in text",
        version: str = "1.0.0",
        model_path: Optional[str] = None,
        toxicity_patterns_path: Optional[str] = None,
        toxicity_threshold: float = 0.7,
        use_llm: bool = True,
        llm_provider: Optional[str] = None,
        llm_config: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize toxicity detection skill.
        
        Args:
            name: Name of the skill
            description: Description of what the skill does
            version: Version of the skill
            model_path: Path to the toxicity detection model
            toxicity_patterns_path: Path to toxicity patterns file
            toxicity_threshold: Threshold above which content is considered toxic
            use_llm: Whether to use LLM for toxicity detection
            llm_provider: Provider for LLM-based detection
            llm_config: Configuration for LLM
        """
        input_schema = {
            "type": "object",
            "properties": {
                "text": {"type": "string", "description": "Text to assess for toxicity"},
                "context": {"type": "string", "description": "Context of the text"},
                "categories": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Specific toxicity categories to assess",
                },
                "metadata": {"type": "object", "description": "Additional metadata"},
            },
            "required": ["text"],
        }
        
        output_schema = {
            "type": "object",
            "properties": {
                "is_toxic": {"type": "boolean", "description": "Whether the content is considered toxic"},
                "toxicity_score": {"type": "number", "description": "Overall toxicity score"},
                "categories": {
                    "type": "object",
                    "description": "Toxicity scores by category",
                },
                "explanations": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Explanations for identified toxicity",
                },
                "suggested_revisions": {
                    "type": "string",
                    "description": "Suggested revision to reduce toxicity",
                },
            },
        }
        
        super().__init__(
            name=name,
            description=description,
            version=version,
            input_schema=input_schema,
            output_schema=output_schema,
            model_path=model_path,
        )
        
        self.toxicity_patterns_path = toxicity_patterns_path
        self.toxicity_threshold = toxicity_threshold
        self.use_llm = use_llm
        self.llm_provider = llm_provider
        self.llm_config = llm_config or {}
        
        self.toxicity_patterns = {}
        self.llm_client = None
    
    def load(self) -> bool:
        """
        Load the toxicity detection model and patterns.
        
        Returns:
            bool: True if loading was successful, False otherwise
        """
        try:
            # Load toxicity patterns if provided
            if self.toxicity_patterns_path and os.path.exists(self.toxicity_patterns_path):
                with open(self.toxicity_patterns_path, 'r') as f:
                    self.toxicity_patterns = json.load(f)
                logger.info(f"Loaded toxicity patterns from {self.toxicity_patterns_path}")
            
            # Set up LLM client if using LLM
            if self.use_llm:
                self._setup_llm_client()
            
            self.loaded = True
            return True
            
        except Exception as e:
            logger.error(f"Failed to load toxicity detection skill: {str(e)}")
            return False
    
    def _setup_llm_client(self):
        """Set up the LLM client based on the provider."""
        if not self.llm_provider:
            # Default to local Ollama model
            from ..reasoning.ollama_engine import OllamaEngine
            self.llm_client = OllamaEngine(
                model=self.llm_config.get("model", "deepseek-r1:14b"),
                base_url=self.llm_config.get("base_url", "http://localhost:11434"),
            )
            logger.info(f"Using Ollama LLM for toxicity detection")
        
        elif self.llm_provider.lower() == "openai":
            try:
                import openai
                self.llm_client = openai.OpenAI(
                    api_key=self.llm_config.get("api_key"),
                    base_url=self.llm_config.get("base_url"),
                )
                logger.info(f"Using OpenAI for toxicity detection")
            except ImportError:
                logger.warning("OpenAI package not found, falling back to Ollama")
                self._setup_llm_client()
        
        elif self.llm_provider.lower() == "anthropic":
            try:
                import anthropic
                self.llm_client = anthropic.Anthropic(
                    api_key=self.llm_config.get("api_key"),
                )
                logger.info(f"Using Anthropic for toxicity detection")
            except ImportError:
                logger.warning("Anthropic package not found, falling back to Ollama")
                self._setup_llm_client()
        
        else:
            logger.warning(f"Unsupported LLM provider: {self.llm_provider}, falling back to Ollama")
            self._setup_llm_client()
    
    def _execute(self, input_data: Dict[str, Any]) -> Tuple[Dict[str, Any], float]:
        """
        Execute toxicity detection on the input text.
        
        Args:
            input_data: Input data containing text to assess
            
        Returns:
            Tuple[Dict[str, Any], float]: Detection results and confidence
        """
        text = input_data["text"]
        context = input_data.get("context", "")
        categories = input_data.get("categories", self.TOXICITY_CATEGORIES)
        
        # Choose the appropriate detection method
        if self.use_llm and self.llm_client:
            result = self._detect_with_llm(text, context, categories)
        else:
            result = self._detect_with_patterns(text, categories)
        
        # Determine if content is toxic based on threshold
        toxicity_score = result["toxicity_score"]
        is_toxic = toxicity_score >= self.toxicity_threshold
        result["is_toxic"] = is_toxic
        
        # Calculate confidence based on the method and results
        if self.use_llm:
            # Higher confidence for LLM-based detection with more explanations
            confidence = 0.7 + min(0.2, len(result["explanations"]) * 0.05)
        else:
            # Pattern-based confidence grows with the number of patterns matched
            confidence = 0.5 + min(0.3, sum(1 for c in result["categories"].values() if c > 0) * 0.05)
        
        return result, confidence
    
    def _detect_with_patterns(self, text: str, categories: List[str]) -> Dict[str, Any]:
        """
        Detect toxicity using pattern matching.
        
        Args:
            text: Text to assess
            categories: Toxicity categories to assess
            
        Returns:
            Dict[str, Any]: Detection results
        """
        # Convert text to lowercase for case-insensitive matching
        lowercase_text = text.lower()
        
        # Initialize category scores
        category_scores = {category: 0.0 for category in categories}
        explanations = []
        
        # Check patterns for each category
        for category in categories:
            if category in self.toxicity_patterns:
                patterns = self.toxicity_patterns[category]
                for pattern_info in patterns:
                    pattern = pattern_info["pattern"]
                    severity = pattern_info.get("severity", 0.5)
                    explanation = pattern_info.get("explanation", f"Matched pattern related to {category}")
                    
                    # Check if pattern matches
                    matches = re.findall(pattern, lowercase_text, re.IGNORECASE)
                    if matches:
                        category_scores[category] += severity * len(matches)
                        explanations.append(f"{category.replace('_', ' ').title()}: {explanation}")
        
        # Normalize category scores to range 0-1
        for category in category_scores:
            if category_scores[category] > 1.0:
                category_scores[category] = 1.0
        
        # Calculate overall toxicity score as the max of category scores
        overall_score = max(category_scores.values()) if category_scores else 0.0
        
        # Generate a suggested revision if content is toxic
        suggested_revision = ""
        if overall_score >= self.toxicity_threshold:
            suggested_revision = self._generate_suggested_revision(text, category_scores)
        
        return {
            "toxicity_score": overall_score,
            "categories": category_scores,
            "explanations": explanations,
            "suggested_revision": suggested_revision,
        }
    
    def _detect_with_llm(self, text: str, context: str, categories: List[str]) -> Dict[str, Any]:
        """
        Detect toxicity using LLM.
        
        Args:
            text: Text to assess
            context: Context of the text
            categories: Toxicity categories to assess
            
        Returns:
            Dict[str, Any]: Detection results
        """
        try:
            # Create prompt for LLM
            categories_str = ", ".join([cat.replace("_", " ") for cat in categories])
            prompt = f"""
            You are a toxicity detection system. Analyze the following text for potential toxicity in these categories: {categories_str}.
            
            Text to analyze: "{text}"
            
            Context (if available): "{context}"
            
            For each toxicity category, assign a score between 0.0 (not toxic) and 1.0 (highly toxic).
            Provide explanations for any identified toxicity.
            If the text is toxic (overall score >= {self.toxicity_threshold}), suggest a revised version that maintains the original meaning but removes the toxic elements.
            
            Format your response as JSON with these fields:
            - toxicity_score: overall toxicity score (0.0-1.0)
            - categories: object with toxicity scores for each category
            - explanations: array of explanations for identified toxicity
            - suggested_revision: suggested revision of the text with toxicity removed
            
            Only include the JSON output, nothing else.
            """
            
            # Use the appropriate LLM client
            response = self._get_llm_response(prompt)
            
            # Parse the response as JSON
            try:
                result = json.loads(response)
                # Ensure all required fields are present
                required_fields = ["toxicity_score", "categories", "explanations", "suggested_revision"]
                for field in required_fields:
                    if field not in result:
                        result[field] = [] if field == "explanations" else (
                            {} if field == "categories" else (
                                "" if field == "suggested_revision" else 0.0
                            )
                        )
                return result
            except json.JSONDecodeError:
                # If parsing fails, extract JSON-like content using regex
                json_pattern = r'({[\s\S]*})'
                match = re.search(json_pattern, response)
                if match:
                    try:
                        result = json.loads(match.group(1))
                        return result
                    except json.JSONDecodeError:
                        pass
                
                # If all parsing attempts fail, return a minimal result
                logger.warning(f"Failed to parse LLM response as JSON: {response[:100]}...")
                return {
                    "toxicity_score": 0.0,  # Default to non-toxic
                    "categories": {category: 0.0 for category in categories},
                    "explanations": ["Failed to properly analyze toxicity."],
                    "suggested_revision": "",
                }
        
        except Exception as e:
            logger.exception(f"Error during LLM-based toxicity detection: {str(e)}")
            return {
                "toxicity_score": 0.0,  # Default to non-toxic
                "categories": {category: 0.0 for category in categories},
                "explanations": [f"Error during toxicity detection: {str(e)}"],
                "suggested_revision": "",
            }
    
    def _get_llm_response(self, prompt: str) -> str:
        """
        Get response from LLM based on the client type.
        
        Args:
            prompt: Prompt for the LLM
            
        Returns:
            str: LLM response
        """
        if hasattr(self.llm_client, "complete"):
            # Ollama-style interface
            response = self.llm_client.complete(prompt)
            return response
        
        elif hasattr(self.llm_client, "chat") and hasattr(self.llm_client.chat, "completions"):
            # OpenAI-style interface
            response = self.llm_client.chat.completions.create(
                model=self.llm_config.get("model", "gpt-4"),
                messages=[{"role": "system", "content": "You are a toxicity detection expert."}, 
                          {"role": "user", "content": prompt}],
                temperature=0.1,
            )
            return response.choices[0].message.content
        
        elif hasattr(self.llm_client, "messages"):
            # Anthropic-style interface
            response = self.llm_client.messages.create(
                model=self.llm_config.get("model", "claude-3-opus-20240229"),
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
            )
            return response.content[0].text
        
        else:
            logger.error(f"Unsupported LLM client interface: {type(self.llm_client)}")
            return "{}"
    
    def _generate_suggested_revision(self, text: str, category_scores: Dict[str, float]) -> str:
        """
        Generate a suggested revision for toxic content.
        
        This uses pattern-based simple approaches. For more complex revisions,
        it's better to use the LLM-based detection which provides revisions directly.
        
        Args:
            text: Original text
            category_scores: Toxicity scores by category
            
        Returns:
            str: Suggested revision or empty string if no revision is needed
        """
        # If no toxicity detected, return original text
        if max(category_scores.values()) < self.toxicity_threshold:
            return ""
        
        # For pattern-based revision, we'll use a simple approach:
        # If we don't have an LLM, we'll just provide a placeholder
        if not self.use_llm or not self.llm_client:
            return "Content flagged as potentially inappropriate. Please consider revising."
        
        # If we have an LLM but somehow didn't use it for detection,
        # let's use it now for revision
        prompt = f"""
        The following text has been flagged as potentially toxic or inappropriate:
        
        "{text}"
        
        Please provide a revised version that maintains the core meaning but removes inappropriate language or content.
        Provide ONLY the revised text, nothing else.
        """
        
        revised_text = self._get_llm_response(prompt)
        
        # Clean up any possible markdown or quotes that the LLM might add
        revised_text = revised_text.strip('`"\'')
        
        return revised_text 