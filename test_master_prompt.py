import sys
import os
import json
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))
from src.pipeline import analyze_conversation

scenarios = [
    {
        "name": "Scenario 1: Delivery Delay (Implicit)",
        "text": """Customer: Where is my order? It has been a week since I paid.
Agent: I am checking your details. Please wait.
Customer: This is frustrating. I need it for work."""
    },
    {
        "name": "Scenario 2: Refund with Severity",
        "text": """Customer: I want a refund now. You rejected it yesterday.
Agent: I understand your concern. Let me know if I can help.
Customer: This is the second time I am asking. Do not play games."""
    },
    {
        "name": "Scenario 3: Product Defect",
        "text": """Customer: My Microsoft Surface Pro is not turning on.
Agent: Have you tried charging it?
Customer: Yes, it is dead. It stopped working this morning."""
    },
    {
        "name": "Scenario 4: Polite Inquiry (Low Severity)",
        "text": """Customer: Hi, does the new laptop come with a warranty?
Agent: Yes, it comes with 1 year warranty.
Customer: Thanks!"""
    },
    {
        "name": "Scenario 5: Neutral Technical Issue (False Complaint check)",
        "text": """Customer: I’m unable to log into my Netflix account since this morning.
Customer: It keeps saying invalid credentials.
Agent: Thanks for reaching out. Please try resetting your password.
Agent: Let me know if the issue continues."""
    }
]

output_file = os.path.join(os.path.dirname(__file__), "final_test_output.txt")
with open(output_file, "w", encoding="utf-8") as f:
    for sc in scenarios:
        f.write(f"\n--- Running {sc['name']} ---\n")
        res = analyze_conversation(sc['text'])
        summary = res['summary']
        f.write(f"Intent: {summary['primary_intent']} (Conf: {summary['intent_confidence']})\n")
        f.write(f"Sentiment: {summary['overall_sentiment']} (Conf: {summary['sentiment_confidence']})\n")
        f.write(f"Empathy Score: {summary['empathy_score']}\n")
        f.write(f"Reason: {summary['intent_reason']}\n")
        f.write(f"Products: {summary['products_detected']}\n")
print(f"Done. Output written to {output_file}")
