"""
Compliance monitoring cognitive skill for agentic systems.

This skill monitors agent outputs for compliance with regulatory requirements,
internal policies, and ethical guidelines.
"""

import os
import json
import logging
import re
from typing import Any, Dict, List, Optional, Tuple, Union
import time

from ..cognitive_skills.base_skill import BaseCognitiveSkill, SkillResponse

logger = logging.getLogger(__name__)

class ComplianceMonitoringSkill(BaseCognitiveSkill):
    """
    Cognitive skill for monitoring compliance with regulatory, legal, and policy requirements.
    
    This skill evaluates text for compliance with various regulatory frameworks,
    industry standards, internal policies, and ethical guidelines. It helps
    ensure that agent outputs meet necessary compliance requirements.
    """
    
    COMPLIANCE_CATEGORIES = [
        "data_privacy",      # GDPR, CCPA, etc.
        "financial",         # Financial regulations (SOX, etc.)
        "healthcare",        # HIPAA, etc.
        "copyright",         # Copyright compliance
        "disclosure",        # Required disclosures
        "accessibility",     # Accessibility requirements
        "safety",            # Safety regulations
        "ethical",           # Ethical guidelines
        "terms_of_service",  # TOS compliance
    ]
    
    def __init__(
        self,
        name: str = "compliance_monitoring",
        description: str = "Monitors compliance with regulatory and policy requirements",
        version: str = "1.0.0",
        model_path: Optional[str] = None,
        compliance_rules_path: Optional[str] = None,
        policy_documents_path: Optional[str] = None,
        compliance_threshold: float = 0.7,
        use_llm: bool = True,
        llm_provider: Optional[str] = None,
        llm_config: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize compliance monitoring skill.
        
        Args:
            name: Name of the skill
            description: Description of what the skill does
            version: Version of the skill
            model_path: Path to the compliance model
            compliance_rules_path: Path to compliance rules file
            policy_documents_path: Path to policy documents directory
            compliance_threshold: Threshold above which content is considered compliant
            use_llm: Whether to use LLM for compliance checking
            llm_provider: Provider for LLM-based checking
            llm_config: Configuration for LLM
        """
        input_schema = {
            "type": "object",
            "properties": {
                "text": {"type": "string", "description": "Text to check for compliance"},
                "context": {"type": "string", "description": "Context of the text"},
                "categories": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Specific compliance categories to check",
                },
                "domain": {"type": "string", "description": "Domain/industry context (e.g., healthcare, finance)"},
                "metadata": {"type": "object", "description": "Additional metadata"},
            },
            "required": ["text"],
        }
        
        output_schema = {
            "type": "object",
            "properties": {
                "is_compliant": {"type": "boolean", "description": "Whether the content is compliant"},
                "compliance_score": {"type": "number", "description": "Overall compliance score"},
                "categories": {
                    "type": "object",
                    "description": "Compliance scores by category",
                },
                "violations": {
                    "type": "array",
                    "items": {"type": "object"},
                    "description": "Compliance violations detected",
                },
                "recommendations": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Recommendations to address compliance issues",
                },
                "suggested_revision": {
                    "type": "string",
                    "description": "Suggested compliant revision",
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
        
        self.compliance_rules_path = compliance_rules_path
        self.policy_documents_path = policy_documents_path
        self.compliance_threshold = compliance_threshold
        self.use_llm = use_llm
        self.llm_provider = llm_provider
        self.llm_config = llm_config or {}
        
        self.compliance_rules = {}
        self.policy_documents = {}
        self.llm_client = None
    
    def load(self) -> bool:
        """
        Load the compliance rules and policy documents.
        
        Returns:
            bool: True if loading was successful, False otherwise
        """
        try:
            # Load compliance rules if provided
            if self.compliance_rules_path and os.path.exists(self.compliance_rules_path):
                with open(self.compliance_rules_path, 'r') as f:
                    self.compliance_rules = json.load(f)
                logger.info(f"Loaded compliance rules from {self.compliance_rules_path}")
            
            # Load policy documents if provided
            if self.policy_documents_path and os.path.exists(self.policy_documents_path):
                for filename in os.listdir(self.policy_documents_path):
                    if filename.endswith('.json'):
                        file_path = os.path.join(self.policy_documents_path, filename)
                        with open(file_path, 'r') as f:
                            policy_name = os.path.splitext(filename)[0]
                            self.policy_documents[policy_name] = json.load(f)
                logger.info(f"Loaded {len(self.policy_documents)} policy documents from {self.policy_documents_path}")
            
            # Set up LLM client if using LLM
            if self.use_llm:
                self._setup_llm_client()
            
            self.loaded = True
            return True
            
        except Exception as e:
            logger.error(f"Failed to load compliance monitoring skill: {str(e)}")
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
            logger.info(f"Using Ollama LLM for compliance monitoring")
        
        elif self.llm_provider.lower() == "openai":
            try:
                import openai
                self.llm_client = openai.OpenAI(
                    api_key=self.llm_config.get("api_key"),
                    base_url=self.llm_config.get("base_url"),
                )
                logger.info(f"Using OpenAI for compliance monitoring")
            except ImportError:
                logger.warning("OpenAI package not found, falling back to Ollama")
                self._setup_llm_client()
        
        elif self.llm_provider.lower() == "anthropic":
            try:
                import anthropic
                self.llm_client = anthropic.Anthropic(
                    api_key=self.llm_config.get("api_key"),
                )
                logger.info(f"Using Anthropic for compliance monitoring")
            except ImportError:
                logger.warning("Anthropic package not found, falling back to Ollama")
                self._setup_llm_client()
        
        else:
            logger.warning(f"Unsupported LLM provider: {self.llm_provider}, falling back to Ollama")
            self._setup_llm_client()
    
    def _execute(self, input_data: Dict[str, Any]) -> Tuple[Dict[str, Any], float]:
        """
        Execute compliance check on the input text.
        
        Args:
            input_data: Input data containing text to assess
            
        Returns:
            Tuple[Dict[str, Any], float]: Compliance check results and confidence
        """
        text = input_data["text"]
        context = input_data.get("context", "")
        categories = input_data.get("categories", self.COMPLIANCE_CATEGORIES)
        domain = input_data.get("domain", "general")
        
        # Choose the appropriate compliance check method
        if self.use_llm and self.llm_client:
            result = self._check_with_llm(text, context, categories, domain)
        else:
            result = self._check_with_rules(text, categories, domain)
        
        # Determine if content is compliant based on threshold
        compliance_score = result["compliance_score"]
        is_compliant = compliance_score >= self.compliance_threshold
        result["is_compliant"] = is_compliant
        
        # Calculate confidence based on the method and results
        if self.use_llm:
            # Higher confidence for LLM-based checking with more recommendations
            confidence = 0.7 + min(0.2, len(result["recommendations"]) * 0.05)
        else:
            # Rule-based confidence grows with the number of rules checked
            confidence = 0.5 + min(0.3, len(result["violations"]) * 0.05)
        
        return result, confidence
    
    def _check_with_rules(self, text: str, categories: List[str], domain: str) -> Dict[str, Any]:
        """
        Check compliance using defined rules.
        
        Args:
            text: Text to check
            categories: Compliance categories to check
            domain: Domain/industry context
            
        Returns:
            Dict[str, Any]: Compliance check results
        """
        # Convert text to lowercase for case-insensitive matching
        lowercase_text = text.lower()
        
        # Initialize category scores and violations
        category_scores = {category: 1.0 for category in categories}  # Start with perfect compliance
        violations = []
        
        # Check rules for each category
        for category in categories:
            if category in self.compliance_rules:
                rules = self.compliance_rules[category]
                for rule in rules:
                    # Check if rule applies to this domain
                    if "domains" in rule and domain not in rule["domains"]:
                        continue
                    
                    rule_id = rule.get("id", f"{category}-rule")
                    description = rule.get("description", "")
                    pattern = rule.get("pattern", "")
                    severity = rule.get("severity", 0.5)
                    
                    # If no pattern is provided, skip this rule
                    if not pattern:
                        continue
                    
                    # Check if pattern matches
                    matches = re.findall(pattern, lowercase_text, re.IGNORECASE)
                    if matches:
                        # Deduct from compliance score based on severity
                        category_scores[category] -= severity
                        
                        # Add violation
                        violations.append({
                            "rule_id": rule_id,
                            "category": category,
                            "description": description,
                            "severity": severity,
                            "matches": matches[:5],  # Limit to first 5 matches
                        })
        
        # Ensure scores are in the range 0.0-1.0
        for category in category_scores:
            if category_scores[category] < 0.0:
                category_scores[category] = 0.0
        
        # Calculate overall compliance score as the minimum of category scores
        overall_score = min(category_scores.values()) if category_scores else 0.0
        
        # Generate recommendations and suggested revision
        recommendations = self._generate_recommendations(violations)
        suggested_revision = ""
        
        return {
            "compliance_score": overall_score,
            "categories": category_scores,
            "violations": violations,
            "recommendations": recommendations,
            "suggested_revision": suggested_revision,
        }
    
    def _check_with_llm(self, text: str, context: str, categories: List[str], domain: str) -> Dict[str, Any]:
        """
        Check compliance using LLM.
        
        Args:
            text: Text to check
            context: Context of the text
            categories: Compliance categories to check
            domain: Domain/industry context
            
        Returns:
            Dict[str, Any]: Compliance check results
        """
        try:
            # Create prompt for LLM
            categories_str = ", ".join([cat.replace("_", " ") for cat in categories])
            
            # Include relevant policy documents if available
            policy_context = ""
            if domain in self.policy_documents:
                policy_context = f"""
                Consider the following policy information for {domain}:
                {json.dumps(self.policy_documents[domain], indent=2)}
                """
            
            prompt = f"""
            You are a compliance expert specializing in {domain} regulations and policies.
            
            Analyze the following text for compliance with these categories: {categories_str}.
            
            Text to analyze: "{text}"
            
            Context (if available): "{context}"
            
            {policy_context}
            
            For each compliance category, assign a score between 0.0 (non-compliant) and 1.0 (fully compliant).
            Identify any compliance violations with specific references to regulations or policies.
            Provide recommendations to address compliance issues.
            If the text is non-compliant (overall score < {self.compliance_threshold}), suggest a revised version.
            
            Format your response as JSON with these fields:
            - compliance_score: overall compliance score (0.0-1.0)
            - categories: object with compliance scores for each category
            - violations: array of objects with rule_id, category, description, and severity for each violation
            - recommendations: array of recommendations to address compliance issues
            - suggested_revision: suggested compliant revision of the text
            
            Only include the JSON output, nothing else.
            """
            
            # Use the appropriate LLM client
            response = self._get_llm_response(prompt)
            
            # Parse the response as JSON
            try:
                result = json.loads(response)
                # Ensure all required fields are present
                required_fields = ["compliance_score", "categories", "violations", "recommendations", "suggested_revision"]
                for field in required_fields:
                    if field not in result:
                        result[field] = [] if field in ["violations", "recommendations"] else (
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
                    "compliance_score": 0.5,  # Default to medium compliance
                    "categories": {category: 0.5 for category in categories},
                    "violations": [],
                    "recommendations": ["Failed to properly analyze compliance. Manual review recommended."],
                    "suggested_revision": "",
                }
        
        except Exception as e:
            logger.exception(f"Error during LLM-based compliance check: {str(e)}")
            return {
                "compliance_score": 0.5,  # Default to medium compliance
                "categories": {category: 0.5 for category in categories},
                "violations": [],
                "recommendations": [f"Error during compliance check: {str(e)}. Manual review recommended."],
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
                messages=[{"role": "system", "content": "You are a compliance monitoring expert."}, 
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
    
    def _generate_recommendations(self, violations: List[Dict[str, Any]]) -> List[str]:
        """
        Generate recommendations based on compliance violations.
        
        Args:
            violations: List of compliance violations
            
        Returns:
            List[str]: List of recommendations
        """
        recommendations = []
        
        # Group violations by category
        violations_by_category = {}
        for violation in violations:
            category = violation.get("category", "unknown")
            if category not in violations_by_category:
                violations_by_category[category] = []
            violations_by_category[category].append(violation)
        
        # Generate recommendations for each category
        for category, category_violations in violations_by_category.items():
            # Get high severity violations
            high_severity = [v for v in category_violations if v.get("severity", 0) >= 0.7]
            
            if high_severity:
                category_name = category.replace("_", " ").title()
                recommendations.append(f"Address critical {category_name} compliance issues.")
                
                # Add specific recommendations based on violation descriptions
                for violation in high_severity[:3]:  # Limit to top 3 high severity violations
                    description = violation.get("description", "")
                    if description:
                        recommendations.append(f"- {description}")
            
            # Add general recommendation for the category if it has violations
            if category_violations:
                if category == "data_privacy":
                    recommendations.append("Ensure all personal data handling complies with relevant privacy regulations.")
                elif category == "financial":
                    recommendations.append("Review content for compliance with financial regulations and disclosure requirements.")
                elif category == "healthcare":
                    recommendations.append("Ensure content complies with healthcare information privacy and security requirements.")
                elif category == "copyright":
                    recommendations.append("Verify that all content is original or properly attributed and licensed.")
                elif category == "disclosure":
                    recommendations.append("Include necessary disclosures and disclaimers.")
                elif category == "accessibility":
                    recommendations.append("Improve accessibility to ensure content is available to all users.")
                elif category == "safety":
                    recommendations.append("Address safety concerns in the content.")
                elif category == "ethical":
                    recommendations.append("Review content against ethical guidelines and standards.")
                elif category == "terms_of_service":
                    recommendations.append("Ensure content complies with platform terms of service.")
        
        # Add general recommendation if no specific ones were added
        if not recommendations:
            recommendations.append("No significant compliance issues detected. Regular compliance monitoring is still recommended.")
        
        return recommendations 