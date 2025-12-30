import json
import os
import logging
from collections import Counter

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

from ner import extract_entities
from sentiment import analyze_sentiment
from intent import detect_intent
from empathy import detect_empathy
from product import detect_products


def refine_intent(text, base_intent):
    """
    Refines intent based on specific keywords as per Business Rules.
    Rules: Refund -> Refund Issue, Late/Deliver -> Delivery Delay, etc.
    """
    text_lower = text.lower()
    
    if "refund" in text_lower or "return" in text_lower or "money back" in text_lower:
        return "Refund Issue"
    
    if "deliver" in text_lower or "arrive" in text_lower or "late" in text_lower or "wait" in text_lower:
        if "not" in text_lower or "hasn't" in text_lower or "delayed" in text_lower:
            return "Delivery Delay"
            
    if "payment" in text_lower or "deducted" in text_lower or "charge" in text_lower:
        return "Payment Issue"
        
    if "damaged" in text_lower or "broken" in text_lower or "not working" in text_lower:
        return "Product Defect"

    # Fallback to the ML-detected intent or map generic ones
    if base_intent == "Complaint":
        # If ML said complaint but we didn't match specific keywords, keep it 'Complaint'
        # or try to find a reason.
        return "Complaint"
        
    return base_intent


def generate_reasoning(overall_sentiment, primary_intent, queries):
    """
    Generates context-aware reasoning strings.
    """
    sentiment_reason = f"Customer expressions indicate {overall_sentiment.lower()} sentiment."
    if overall_sentiment == "Negative":
        sentiment_reason = "Customer reports issues such as delivery delays, refund rejections, or product defects, indicating frustration."
    elif overall_sentiment == "Positive":
        sentiment_reason = "Customer expressed satisfaction with the service or product."

    intent_reason = f"The primary interaction revolves around {primary_intent}."
    if primary_intent == "Delivery Delay":
        intent_reason = "The customer is specifically inquiring about a delayed order that has not arrived."
    elif primary_intent == "Refund Issue":
        intent_reason = "The interaction is focused on a request for a refund or return of a product."
    elif primary_intent == "Payment Issue":
        intent_reason = "The customer reported a payment deduction without service activation."

    return sentiment_reason, intent_reason


def analyze_conversation(text):
    logger.info("Starting analysis with Master Prompt Rules...")
    lines = text.split('\n')
    
    # storage
    all_entities = []
    customer_lines = []
    agent_lines = []
    
    all_customer_intents = []
    conversation_sentiment_scores = []
    agent_empathy_scores = []
    
    # 1. Process Line-by-Line
    for line in lines:
        if not line.strip():
            continue
            
        speaker = "Unknown"
        content = line
        if ":" in line:
            parts = line.split(":", 1)
            speaker = parts[0].strip()
            content = parts[1].strip()
            
        # Extract Entities everywhere
        ents = extract_entities(content)
        all_entities.extend(ents)
        
        if speaker.lower() == "customer":
            customer_lines.append(content)
            
            # Sentiment (Context)
            sent = analyze_sentiment(content)
            conversation_sentiment_scores.append(sent['polarity'])
            
            # Intent (ML + Rule Refinement)
            base_intent = detect_intent(content)
            refined = refine_intent(content, base_intent)
            all_customer_intents.append(refined)
            
        elif speaker.lower() == "agent":
            agent_lines.append(content)
            
            # Empathy (Agent only)
            emp = detect_empathy(content)
            if emp['detected']:
                agent_empathy_scores.append(emp['confidence'])
            else:
                 agent_empathy_scores.append(0.0)

    # 2. Aggregations & Logic enforcing
    
    # Primary Intent
    if all_customer_intents:
        # Prioritize "Severe" intents if present
        severity_order = ["Refund Issue", "Payment Issue", "Product Defect", "Delivery Delay", "Complaint", "Support Request", "Inquiry", "Feedback"]
        primary_intent = "General"
        
        # Find the most severe intent present
        found_severe = False
        for high_intent in severity_order:
            if high_intent in all_customer_intents:
                primary_intent = high_intent
                found_severe = True
                break
        
        if not found_severe:
             # Fallback to most frequent
             primary_intent = max(set(all_customer_intents), key=all_customer_intents.count)
    else:
        primary_intent = "Unknown"

    # Overall Sentiment (Rule: Complaints/Issues MUST be Negative)
    negative_intents = ["Refund Issue", "Payment Issue", "Product Defect", "Delivery Delay", "Complaint"]
    
    if primary_intent in negative_intents:
        overall_sentiment = "Negative"
    else:
        # Fallback to polarity average
        avg_polarity = sum(conversation_sentiment_scores) / len(conversation_sentiment_scores) if conversation_sentiment_scores else 0
        if avg_polarity > 0.1: overall_sentiment = "Positive"
        elif avg_polarity < -0.1: overall_sentiment = "Negative"
        else: overall_sentiment = "Neutral"

    # Empathy Score
    empathy_score = 0.0
    if agent_empathy_scores:
        empathy_score = sum(agent_empathy_scores) / len(agent_empathy_scores)
    
    # Reasoning
    sentiment_reason, intent_reason = generate_reasoning(overall_sentiment, primary_intent, customer_lines)
    
    # Products
    products = detect_products(text, all_entities)
    
    # Unique entities
    unique_entities = {f"{e['text']}_{e['label']}": e for e in all_entities}.values()

    # 3. Construct Final JSON (Schema Compliance)
    result = {
        "summary": {
            "overall_sentiment": overall_sentiment,
            "sentiment_reason": sentiment_reason,
            "primary_intent": primary_intent,
            "intent_reason": intent_reason,
            "empathy_score": round(empathy_score, 2),
            "products_detected": sorted(products),
            "entity_count": len(all_entities) # Prompt asked for total count of relevant entities found
        },
        "customer_queries": customer_lines,
        "agent_responses": agent_lines,
        "details": {
            "intents_found": list(set(all_customer_intents)),
            "entities": list(unique_entities)
        }
    }
    
    logger.info("Analysis complete.")
    return result


if __name__ == "__main__":
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    input_path = os.path.join(BASE_DIR, "data", "sample_conversation.txt")
    output_dir = os.path.join(BASE_DIR, "outputs")
    output_path = os.path.join(output_dir, "analysis_result.json")

    os.makedirs(output_dir, exist_ok=True)

    if not os.path.exists(input_path):
        logger.error(f"Input file not found: {input_path}")
        exit(1)

    with open(input_path, "r", encoding="utf-8") as file:
        conversation = file.read().strip()

    if not conversation:
        logger.error("Empty conversation file.")
        exit(1)

    analysis = analyze_conversation(conversation)

    with open(output_path, "w", encoding="utf-8") as out:
        json.dump(analysis, out, indent=4)

    print(f"✅ Processing Complete. Results saved to: {output_path}")
