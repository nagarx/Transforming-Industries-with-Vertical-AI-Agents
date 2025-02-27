# Customer Support HITL Agent Example

This example demonstrates a Human-in-the-Loop (HITL) agent implementation for customer support using the deepseek-r1:14b local model. The agent handles customer inquiries while requesting human validation for sensitive operations or when confidence is low.

## Key Features

- **Human-in-the-Loop Approach**: Requests human feedback for sensitive operations
- **Risk Assessment**: Determines when human oversight is necessary
- **Memory Functionality**: Maintains conversation context
- **Customer Data Integration**: Incorporates customer information into agent context
- **Product Knowledge**: Utilizes product information to answer customer inquiries

## Running the Example

To run this example, follow these steps:

1. Make sure you have Ollama installed and the deepseek-r1:14b model pulled:
   ```
   ollama pull deepseek-r1:14b
   ```

2. Run the interactive demo:
   ```
   cd agentic-systems
   python examples/customer_support/hitl_customer_agent.py
   ```

3. Follow the prompts to select a customer and start asking questions.

## Example Queries

You can try queries such as:
- "What subscription plans do you offer?"
- "I want to cancel my subscription"
- "How do I reset my password?"
- "Can I get a refund for my last payment?"

The agent will request human feedback for sensitive queries (like cancellations or refunds) while answering general information queries directly.

## Implementation Details

This example demonstrates several key components of the agentic systems framework:

1. **HITLAgent**: Human-augmented agent implementation that requests feedback when needed
2. **OllamaReasoning**: Integration with the local deepseek-r1:14b model
3. **RiskAssessmentSkill**: Evaluation of risk in customer interactions
4. **ShortTermMemory**: Maintaining conversation context

The implementation shows how human oversight can be integrated into an autonomous agent workflow, ensuring that sensitive operations receive appropriate validation while allowing the agent to handle routine inquiries independently. 