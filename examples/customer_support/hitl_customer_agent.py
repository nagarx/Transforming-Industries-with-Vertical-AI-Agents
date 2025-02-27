"""
Human-in-the-Loop Customer Support Agent Example

This example demonstrates a customer support agent that uses human validation
for important decisions, using the deepseek-r1:14b local model.
"""

import os
import sys
import logging
from typing import Dict, Any, List

# Add the parent directory to the path to import agentic_systems
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from agentic_systems.agents import HITLAgent, HumanFeedback, HumanFeedbackType
from agentic_systems.core.reasoning import OllamaReasoning
from agentic_systems.core.memory import ShortTermMemory
from agentic_systems.core.cognitive_skills import RiskAssessmentSkill

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Sample customer data (in a real application, this would come from a database)
CUSTOMER_DATA = {
    "customer123": {
        "name": "Alice Johnson",
        "email": "alice@example.com",
        "subscription": "Premium",
        "billing_cycle": "Monthly",
        "last_payment": "2023-01-15",
        "support_history": [
            {"date": "2023-01-05", "issue": "Login problem", "resolved": True},
            {"date": "2023-01-10", "issue": "Billing question", "resolved": True},
        ]
    },
    "customer456": {
        "name": "Bob Smith",
        "email": "bob@example.com",
        "subscription": "Basic",
        "billing_cycle": "Annual",
        "last_payment": "2022-12-01",
        "support_history": [
            {"date": "2022-12-15", "issue": "Feature request", "resolved": True},
        ]
    }
}

# Sample product knowledge base
PRODUCT_KNOWLEDGE = [
    "Our subscription service has three tiers: Basic, Standard, and Premium.",
    "Premium subscribers get 24/7 priority support and access to all features.",
    "Basic subscription costs $9.99/month or $99/year.",
    "Standard subscription costs $19.99/month or $199/year.",
    "Premium subscription costs $29.99/month or $299/year.",
    "To upgrade your subscription, go to Account Settings > Subscription.",
    "We offer a 30-day money-back guarantee for all subscription tiers.",
    "Annual billing gives you a 20% discount compared to monthly billing.",
    "Account passwords must be reset through the 'Forgot Password' link on the login page.",
    "Our API documentation is available at docs.example.com/api."
]

class CustomerSupportAgent:
    """Customer support agent using a Human-in-the-Loop approach."""
    
    def __init__(self):
        """Initialize the customer support agent."""
        # Initialize the reasoning engine with the local Ollama model
        self.reasoning_engine = OllamaReasoning(
            model_name="deepseek-r1:14b",
            base_url="http://localhost:11434",
            temperature=0.5,  # Lower temperature for more predictable responses
            system_prompt_template=(
                "You are an AI customer support agent that helps customers with their issues. "
                "You have access to customer data and product knowledge, but for sensitive operations "
                "like refunds, cancellations, or account changes, you must get human approval. "
                "Always be polite, helpful, and concise. "
                "{context}"
                "{persona}"
            )
        )
        
        # Create the risk assessment skill
        self.risk_assessment = RiskAssessmentSkill(
            name="customer_risk_assessment",
            description="Assesses risk in customer support interactions",
            use_llm=True,
            llm_config={
                "model": "deepseek-r1:14b",
                "base_url": "http://localhost:11434",
            }
        )
        
        # Initialize the HITL agent
        self.agent = HITLAgent(
            agent_id="customer_support_agent",
            name="Customer Support Agent",
            description="An agent that helps customers with support issues, with human oversight for sensitive operations",
            reasoning_engine=self.reasoning_engine,
            human_feedback_fn=HITLAgent.create_console_feedback_fn(),  # Use console for feedback
            memory=ShortTermMemory(max_items=20),  # Remember recent interactions
            cognitive_skills=[self.risk_assessment],
            always_require_feedback=False,  # Only require feedback for sensitive operations
            confidence_threshold=0.8,
            high_stakes_keywords=["refund", "cancel", "upgrade", "downgrade", "password", "credit card", "billing"],
            feedback_required_categories=["account_changes", "billing_operations", "security"],
        )
    
    def _get_customer_context(self, customer_id: str) -> Dict[str, Any]:
        """Get context information for a customer."""
        if customer_id in CUSTOMER_DATA:
            customer = CUSTOMER_DATA[customer_id]
            return {
                "customer_data": customer,
                "category": "customer_support",
                "has_customer_data": True
            }
        else:
            return {
                "category": "customer_support",
                "has_customer_data": False
            }
    
    def _get_knowledge_context(self) -> List[str]:
        """Get product knowledge context."""
        return PRODUCT_KNOWLEDGE
    
    def handle_query(self, customer_id: str, query: str) -> str:
        """
        Handle a customer query.
        
        Args:
            customer_id: ID of the customer
            query: The customer's query
            
        Returns:
            str: Response to the customer
        """
        # Get customer and knowledge context
        context = self._get_customer_context(customer_id)
        knowledge = self._get_knowledge_context()
        
        # Add knowledge to context
        context["product_knowledge"] = knowledge
        
        # Execute the agent
        logger.info(f"Processing query for customer {customer_id}: {query}")
        response = self.agent.execute(query, context)
        
        # Log if human feedback was requested
        if response.metadata.get("human_feedback_requested", False):
            logger.info("Human feedback was requested and incorporated into the response")
        
        return response.content

def interactive_demo():
    """Run an interactive demo of the customer support agent."""
    print("\n===== Customer Support Agent Demo =====")
    print("This demo simulates a customer support interaction using a Human-in-the-Loop agent.")
    print("The agent will request human feedback for sensitive operations.\n")
    
    # Initialize the agent
    agent = CustomerSupportAgent()
    
    # Select a customer
    print("Available customers:")
    for i, customer_id in enumerate(CUSTOMER_DATA.keys(), 1):
        customer = CUSTOMER_DATA[customer_id]
        print(f"{i}. {customer['name']} ({customer_id}) - {customer['subscription']} plan")
    
    selection = input("\nSelect a customer (1-2) or press Enter for anonymous customer: ")
    
    if selection and selection.isdigit() and 1 <= int(selection) <= len(CUSTOMER_DATA):
        customer_id = list(CUSTOMER_DATA.keys())[int(selection) - 1]
        customer_name = CUSTOMER_DATA[customer_id]["name"]
        print(f"\nSelected customer: {customer_name}")
    else:
        customer_id = "anonymous"
        print("\nAnonymous customer selected")
    
    # Interactive chat loop
    print("\nType your customer support queries below. Type 'exit' to end the demo.\n")
    while True:
        query = input("\nCustomer Query: ")
        if query.lower() in ["exit", "quit", "bye"]:
            print("Ending demo. Goodbye!")
            break
        
        response = agent.handle_query(customer_id, query)
        print(f"\nAgent Response: {response}")

if __name__ == "__main__":
    interactive_demo() 