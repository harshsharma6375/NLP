import json
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

from ner import extract_entities
from sentiment import analyze_sentiment
from intent import detect_intent
from empathy import detect_empathy
from product import detect_products


def analyze_conversation(text):
    """
    Main orchestration function.
    """
    logger.info("Starting analysis...")
    lines = text.split('\n')
    
    # Aggregates
    conversation_sentiment_scores = []
    all_customer_intents = []
    agent_empathy_data = []
    all_entities = []
    
    # New: Collect specific queries
    customer_queries = []
    
    for line in lines:
        if not line.strip():
            continue
            
        # 1. Speaker Verification
        speaker = "Unknown"
        content = line
        if ":" in line:
            parts = line.split(":", 1)
            speaker = parts[0].strip()
            content = parts[1].strip()
            
        # 2. Entity Extraction
        entities = extract_entities(content)
        all_entities.extend(entities)
        
        # 3. Sentiment Analysis
        sentiment = analyze_sentiment(content)
        conversation_sentiment_scores.append(sentiment['polarity']) 
        
        # 4. Speaker Specific Logic
        intent = None
        empathy_info = None
        
        if speaker.lower() == "customer":
            intent = detect_intent(content)
            all_customer_intents.append(intent)
            
            # Query Extraction Logic
            # If intent is Inquiry OR text ends with ? OR starts with Wh- word
            is_question = content.strip().endswith("?") or \
                          content.lower().startswith(("what", "how", "when", "where", "why", "can", "could"))
            
            if intent in ["Inquiry", "Feedback", "Complaint", "Support Request"] or is_question:
                customer_queries.append(content)
            
        elif speaker.lower() == "agent":
            empathy_info = detect_empathy(content)
            if empathy_info['detected']:
                agent_empathy_data.append(empathy_info)

    # 5. Product Detection
    products = detect_products(text, all_entities)

    # 6. Final Aggregation
    avg_polarity = sum(conversation_sentiment_scores) / len(conversation_sentiment_scores) if conversation_sentiment_scores else 0
    overall_sentiment = "Neutral"
    if avg_polarity > 0.1: overall_sentiment = "Positive"
    elif avg_polarity < -0.1: overall_sentiment = "Negative"
    
    primary_intent = max(set(all_customer_intents), key=all_customer_intents.count) if all_customer_intents else "General"
    
    empathy_score = 0.0
    if agent_empathy_data:
        empathy_score = sum(e['confidence'] for e in agent_empathy_data) / len(agent_empathy_data)
        
    unique_entities = {f"{e['text']}_{e['label']}": e for e in all_entities}.values()

    result = {
        "summary": {
            "overall_sentiment": overall_sentiment,
            "sentiment_score": round(avg_polarity, 4),
            "primary_intent": primary_intent,
            "empathy_score": round(empathy_score, 4),
            "products_detected": products,
            "entity_count": len(all_entities)
        },
        "customer_queries": customer_queries,
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
