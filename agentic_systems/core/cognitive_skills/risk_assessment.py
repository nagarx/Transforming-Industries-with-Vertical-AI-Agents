"""
Risk assessment cognitive skill for agentic systems.

This skill evaluates the risk level of agent operations, responses, and actions
to help maintain safe and responsible AI behavior.
"""

import os
import json
import logging
from typing import Any, Dict, List, Optional, Tuple, Union
import time
import re

from ..cognitive_skills.base_skill import BaseCognitiveSkill, SkillResponse

logger = logging.getLogger(__name__)

class RiskAssessmentSkill(BaseCognitiveSkill):
    """
    Cognitive skill for assessing risks in agent operations and responses.
    
    This skill evaluates text for potential risks, including security risks,
    harmful content, potential for misuse, and other categories of risk.
    It provides a risk score and categorization of detected risks.
    """
    
    RISK_CATEGORIES = [
        "security",        # Security vulnerabilities or risks
        "harm",            # Potential to cause harm
        "misuse",          # Potential for misuse
        "confidentiality", # Confidentiality risks
        "bias",            # Bias or unfairness
        "legality",        # Legal compliance issues
        "ethics",          # Ethical concerns
        "financial",       # Financial risks
        "reputational",    # Reputational risks
    ]
    
    def __init__(
        self,
        name: str = "risk_assessment",
        description: str = "Assesses risk levels in agent operations and responses",
        version: str = "1.0.0",
        model_path: Optional[str] = None,
        risk_patterns_path: Optional[str] = None,
        risk_threshold: float = 0.7,
        use_llm: bool = True,
        llm_provider: Optional[str] = None,
        llm_config: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize the risk assessment skill.
        
        Args:
            name: Name of the skill
            description: Description of the skill
            version: Version of the skill
            model_path: Path to a local model for risk assessment
            risk_patterns_path: Path to a JSON file containing risk patterns for rule-based assessment
            risk_threshold: Threshold for flagging risks (0-1)
            use_llm: Whether to use an LLM for risk assessment
            llm_provider: Provider for the LLM (e.g., "openai", "anthropic")
            llm_config: Configuration for the LLM
        """
        super().__init__(name, description, version)
        
        self.model_path = model_path
        self.risk_patterns_path = risk_patterns_path
        self.risk_threshold = risk_threshold
        self.use_llm = use_llm
        self.llm_provider = llm_provider
        self.llm_config = llm_config or {}
        
        # Risk patterns for rule-based assessment
        self.risk_patterns = {}
        
        # LLM client for assessment
        self.llm_client = None
        
        # Load the skill
        self.is_loaded = self.load()
    
    def load(self) -> bool:
        """Load the risk assessment skill."""
        try:
            # Load risk patterns if available
            if self.risk_patterns_path and os.path.exists(self.risk_patterns_path):
                with open(self.risk_patterns_path, 'r') as f:
                    self.risk_patterns = json.load(f)
                logger.info(f"Loaded risk patterns from {self.risk_patterns_path}")
            
            # Set up LLM client if needed
            if self.use_llm:
                self._setup_llm_client()
            
            return True
        except Exception as e:
            logger.error(f"Failed to load risk assessment skill: {str(e)}")
            return False
    
    def _setup_llm_client(self):
        """Set up the LLM client based on the provider."""
        if not self.llm_provider:
            # Default to local Ollama model
            from ..reasoning.ollama_reasoning import OllamaReasoning
            self.llm_client = OllamaReasoning(
                model_name=self.llm_config.get("model", "deepseek-r1:14b"),
                base_url=self.llm_config.get("base_url", "http://localhost:11434"),
            )
            logger.info(f"Using Ollama LLM for risk assessment")
        
        elif self.llm_provider.lower() == "openai":
            try:
                import openai
                self.llm_client = openai.OpenAI(
                    api_key=self.llm_config.get("api_key"),
                    base_url=self.llm_config.get("base_url"),
                )
                logger.info(f"Using OpenAI for risk assessment")
            except ImportError:
                logger.warning("OpenAI package not found, falling back to Ollama")
                self._setup_llm_client()
        
        elif self.llm_provider.lower() == "anthropic":
            try:
                import anthropic
                self.llm_client = anthropic.Anthropic(
                    api_key=self.llm_config.get("api_key"),
                )
                logger.info(f"Using Anthropic for risk assessment")
            except ImportError:
                logger.warning("Anthropic package not found, falling back to Ollama")
                self._setup_llm_client()
        
        else:
            logger.warning(f"Unsupported LLM provider: {self.llm_provider}, falling back to Ollama")
            self._setup_llm_client()
    
    def _execute(self, input_data: Dict[str, Any]) -> Tuple[Dict[str, Any], float]:
        """
        Execute risk assessment on the input text.
        
        Args:
            input_data: Input data containing text to assess
            
        Returns:
            Tuple[Dict[str, Any], float]: Assessment results and confidence
        """
        text = input_data["text"]
        context = input_data.get("context", "")
        categories = input_data.get("categories", self.RISK_CATEGORIES)
        
        # Choose the appropriate assessment method
        if self.use_llm and self.llm_client:
            result = self._assess_with_llm(text, context, categories)
        else:
            result = self._assess_with_patterns(text, categories)
        
        # Determine overall risk level
        risk_score = result["risk_score"]
        if risk_score >= self.risk_threshold:
            risk_level = "high"
        elif risk_score >= self.risk_threshold * 0.5:
            risk_level = "medium"
        else:
            risk_level = "low"
        
        result["risk_level"] = risk_level
        
        # Calculate confidence based on the method and results
        if self.use_llm:
            # Higher confidence for LLM-based assessment with more explanations
            confidence = 0.7 + min(0.2, len(result["explanations"]) * 0.05)
        else:
            # Pattern-based confidence grows with the number of patterns matched
            confidence = 0.5 + min(0.3, sum(1 for c in result["categories"].values() if c > 0) * 0.05)
        
        return result, confidence
    
    def _assess_with_patterns(self, text: str, categories: List[str]) -> Dict[str, Any]:
        """
        Assess risk using pattern matching.
        
        Args:
            text: Text to assess
            categories: Risk categories to assess
            
        Returns:
            Dict[str, Any]: Assessment results
        """
        # Convert text to lowercase for case-insensitive matching
        lowercase_text = text.lower()
        
        # Initialize category scores
        category_scores = {category: 0.0 for category in categories}
        explanations = []
        
        # Check patterns for each category
        for category in categories:
            if category in self.risk_patterns:
                patterns = self.risk_patterns[category]
                for pattern_info in patterns:
                    pattern = pattern_info["pattern"]
                    severity = pattern_info.get("severity", 0.5)
                    explanation = pattern_info.get("explanation", f"Matched pattern related to {category} risk")
                    
                    # Check if pattern matches
                    matches = re.findall(pattern, lowercase_text, re.IGNORECASE)
                    if matches:
                        category_scores[category] += severity * len(matches)
                        explanations.append(f"{category.capitalize()} risk: {explanation}")
        
        # Normalize category scores to range 0-1
        for category in category_scores:
            if category_scores[category] > 1.0:
                category_scores[category] = 1.0
        
        # Calculate overall risk score as the max of category scores
        overall_score = max(category_scores.values()) if category_scores else 0.0
        
        # Generate recommendations based on identified risks
        recommendations = self._generate_recommendations(category_scores)
        
        return {
            "risk_score": overall_score,
            "categories": category_scores,
            "explanations": explanations,
            "recommendations": recommendations,
        }
    
    def _assess_with_llm(self, text: str, context: str, categories: List[str]) -> Dict[str, Any]:
        """
        Assess risk using LLM.
        
        Args:
            text: Text to assess
            context: Context of the text
            categories: Risk categories to assess
            
        Returns:
            Dict[str, Any]: Assessment results
        """
        try:
            # Create prompt for LLM
            categories_str = ", ".join(categories)
            prompt = f"""
            You are a risk assessment expert. Analyze the following text for potential risks in these categories: {categories_str}.
            
            Text to analyze: "{text}"
            
            Context (if available): "{context}"
            
            For each risk category, assign a risk score between 0.0 (no risk) and 1.0 (highest risk).
            Provide explanations for any identified risks.
            Suggest recommendations for mitigating the identified risks.
            
            Format your response as JSON with these fields:
            - risk_score: overall risk score (0.0-1.0)
            - categories: object with risk scores for each category
            - explanations: array of explanations for identified risks
            - recommendations: array of recommendations for mitigating risks
            
            Only include the JSON output, nothing else.
            """
            
            # Use the appropriate LLM client
            response = self._get_llm_response(prompt)
            
            # Parse the response as JSON
            try:
                result = json.loads(response)
                # Ensure all required fields are present
                required_fields = ["risk_score", "categories", "explanations", "recommendations"]
                for field in required_fields:
                    if field not in result:
                        result[field] = [] if field in ["explanations", "recommendations"] else (
                            {} if field == "categories" else 0.0
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
                    "risk_score": 0.5,  # Default to medium risk
                    "categories": {category: 0.5 for category in categories},
                    "explanations": ["Failed to properly analyze risk. Using default medium risk level."],
                    "recommendations": ["Manually review the content for risk assessment."],
                }
        
        except Exception as e:
            logger.exception(f"Error during LLM-based risk assessment: {str(e)}")
            return {
                "risk_score": 0.5,  # Default to medium risk
                "categories": {category: 0.5 for category in categories},
                "explanations": [f"Error during risk assessment: {str(e)}"],
                "recommendations": ["Manually review the content for risk assessment."],
            }
    
    def _get_llm_response(self, prompt: str) -> str:
        """
        Get response from LLM.
        
        Args:
            prompt: The prompt to send to the LLM
            
        Returns:
            str: The LLM response
        """
        if hasattr(self.llm_client, "reason"):
            # Our OllamaReasoning interface
            from ..reasoning import ReasoningInput, ReasoningOutput
            
            reasoning_input = ReasoningInput(
                prompt=prompt,
                system_prompt="You are a risk assessment expert. Analyze the text for potential risks."
            )
            
            reasoning_output = self.llm_client.reason(reasoning_input)
            return reasoning_output.response
        
        elif hasattr(self.llm_client, "complete"):
            # Ollama-style interface
            response = self.llm_client.complete(prompt)
            return response
        
        elif hasattr(self.llm_client, "chat") and hasattr(self.llm_client.chat, "completions"):
            # OpenAI-style interface
            response = self.llm_client.chat.completions.create(
                model=self.llm_config.get("model", "gpt-4"),
                messages=[{"role": "system", "content": "You are a risk assessment expert."}, 
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
    
    def _generate_recommendations(self, category_scores: Dict[str, float]) -> List[str]:
        """
        Generate recommendations based on identified risks.
        
        Args:
            category_scores: Risk scores by category
            
        Returns:
            List[str]: List of recommendations
        """
        recommendations = []
        
        # Get high-risk categories
        high_risk_categories = [cat for cat, score in category_scores.items() if score >= self.risk_threshold]
        
        # General recommendation for high risk
        if high_risk_categories:
            recommendations.append(f"High risk detected in {', '.join(high_risk_categories)}. Review content carefully.")
        
        # Specific recommendations by category
        for category, score in category_scores.items():
            if score >= self.risk_threshold:
                if category == "security":
                    recommendations.append("Review for security vulnerabilities and ensure proper protections.")
                elif category == "harm":
                    recommendations.append("Check content for potential harm and modify to reduce risk.")
                elif category == "misuse":
                    recommendations.append("Consider how this content could be misused and add safeguards.")
                elif category == "confidentiality":
                    recommendations.append("Ensure no sensitive or confidential information is being exposed.")
                elif category == "bias":
                    recommendations.append("Examine content for potential bias and ensure fairness.")
                elif category == "legality":
                    recommendations.append("Verify legal compliance of the content and consult legal experts if needed.")
                elif category == "ethics":
                    recommendations.append("Consider ethical implications and ensure alignment with ethical standards.")
                elif category == "financial":
                    recommendations.append("Assess potential financial risks and implement appropriate controls.")
                elif category == "reputational":
                    recommendations.append("Evaluate how this could impact reputation and modify if needed.")
        
        # Add general recommendation if no specific ones were added
        if not recommendations:
            recommendations.append("No significant risks detected. Regular monitoring is still recommended.")
        
        return recommendations 