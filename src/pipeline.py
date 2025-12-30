import json
import os

from ner import extract_entities
from sentiment import analyze_sentiment
from intent import detect_intent
from empathy import detect_empathy
from product import detect_products


def analyze_conversation(text):
    lines = text.split('\n')
    
    # Aggregated results
    all_entities = []
    customer_intents = []
    agent_empathy = False
    details = []

    for line in lines:
        if not line.strip():
            continue
            
        speaker = "Unknown"
        content = line
        
        if ":" in line:
            parts = line.split(":", 1)
            speaker = parts[0].strip()
            content = parts[1].strip()
            
        # Common checks
        sentiment = analyze_sentiment(content)
        entities = extract_entities(content)
        all_entities.extend(entities)
        
        # Specific checks
        intent = None
        is_empathetic = False
        
        if speaker.lower() == "customer":
            intent = detect_intent(content)
            if intent:
                customer_intents.append(intent)
        elif speaker.lower() == "agent":
            is_empathetic = detect_empathy(content)
            if is_empathetic:
                agent_empathy = True
                
        details.append({
            "speaker": speaker,
            "text": content,
            "sentiment": sentiment,
            "intent": intent,
            "empathy_detected": is_empathetic
        })

    # Detect products across all entities and text
    products = detect_products(text, all_entities)

    return {
        "summary": {
            "customer_intent": list(set(customer_intents)),
            "agent_empathy_shown": agent_empathy,
            "products_mentioned": products,
            "entity_count": len(all_entities)
        },
        "details": details
    }


if __name__ == "__main__":

    # Resolve project base directory safely
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    input_path = os.path.join(BASE_DIR, "data", "sample_conversation.txt")
    output_dir = os.path.join(BASE_DIR, "outputs")
    output_path = os.path.join(output_dir, "analysis_result.json")

    # Ensure outputs directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Read input conversation
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input file not found: {input_path}")

    with open(input_path, "r", encoding="utf-8") as file:
        conversation = file.read().strip()

    if not conversation:
        raise ValueError("Input conversation file is empty. Please add conversation text.")

    # Run NLP pipeline
    analysis = analyze_conversation(conversation)

    # Write output JSON
    with open(output_path, "w", encoding="utf-8") as out:
        json.dump(analysis, out, indent=4)

    print("✅ Conversation analysis completed successfully.")
    print(f"📥 Input file  : {input_path}")
    print(f"📤 Output file : {output_path}")
