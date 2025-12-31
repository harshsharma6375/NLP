import json, os, logging
from collections import Counter
from ner import extract_entities
from sentiment import analyze_sentiment
from intent import detect_intent
from empathy import detect_empathy
from product import detect_products

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

def generate_reasoning(sentiment, intent, queries):
    text = " ".join(queries).lower()
    issues = [i for k, i in {
        "delay": "delivery delays", "waiting": "delivery delays", "refund": "payment/refund issues",
        "trust": "loss of trust", "regret": "loss of trust", "working": "product defect", 
        "defective": "product defect", "again": "repeated issues", "second time": "repeated issues"
    }.items() if k in text]
    
    if sentiment == "Negative":
        sent_reason = f"Customer reports {', '.join(set(issues))}, indicating strong negative sentiment." if issues else "Customer expressions indicate frustration and dissatisfaction."
    elif sentiment == "Positive": sent_reason = "Customer uses polite language and expresses satisfaction."
    else: sent_reason = "Interaction is neutral with no strong emotional indicators."

    intent_map = {
        "Complaint": "Customer reports repeated failures, necessitating a formal Complaint." if "repeated" in text else "Strong signals indicate a Complaint.",
        "Product Defect": "Customer reports malfunctioning hardware/software.",
        "Refund Issue": "Customer is disputing a refund rejection or failure.",
        "Delivery Delay": "Customer is inquiring about a late delivery.",
        "Inquiry": "Customer is asking questions without severity."
    }
    return sent_reason, intent_map.get(intent, f"Interaction revolves around {intent}.")

def analyze_conversation(text):
    logger.info("Starting analysis...")
    lines, full_lower = text.split('\n'), text.lower()
    cust_lines, agent_lines, entities, intents, sent_scores, emp_scores = [], [], [], [], [], []

    for line in lines:
        if not line.strip(): continue
        speaker, content = line.split(":", 1) if ":" in line else ("Unknown", line)
        entities.extend(extract_entities(content))
        
        if "customer" in speaker.lower():
            cust_lines.append(content)
            sent_scores.append(analyze_sentiment(content))
            intents.append(detect_intent(content))
        elif "agent" in speaker.lower():
            agent_lines.append(content)
            emp_scores.append(detect_empathy(content))

    # Aggregation
    unique_intents = list({i['label'] for i in intents})
    has_esc = any(k in full_lower for k in ["second time", "again", "wasted", "regret", "lose trust"])
    has_frust = any(k in full_lower for k in ["frustrating", "angry", "disappointed"])
    
    primary, conf = "Inquiry", 0.0
    if has_esc or (has_frust and "working" in full_lower): primary, conf = "Complaint", 0.78
    elif "Product Defect" in unique_intents: primary, conf = "Product Defect", 0.70
    elif unique_intents: primary, conf = unique_intents[0], 0.60
    
    # Confidence update from BERT
    conf = max(conf, max([i['score'] for i in intents if i['label'] == primary], default=0))
    conf = min(max(conf, 0.40), 0.85)
    if primary not in unique_intents: unique_intents.insert(0, primary)

    # Sentiment
    neg = sum(1 for s in sent_scores if s['label'] == "NEGATIVE")
    ov_sent, sent_conf = "Neutral", 0.65
    if has_esc or "regret" in full_lower: ov_sent, sent_conf = "Negative", 0.92
    elif neg > len(sent_scores)/2: ov_sent, sent_conf = "Negative", min(0.9, 0.75 + neg*0.05)
    elif all(s['label'] == "POSITIVE" for s in sent_scores): ov_sent, sent_conf = "Positive", 0.85

    # Empathy
    emp_score, emp_conf = 0.0, 0.0
    if emp_scores:
        emp_score = round(sum(e['score'] for e in emp_scores)/len(emp_scores), 2)
        emp_conf = round(sum(e['confidence'] for e in emp_scores)/len(emp_scores), 2)
        if sum(1 for s in ["sorry", "apologize", "understand"] if s in " ".join(agent_lines).lower()) >= 3:
            emp_score = max(emp_score, 0.85)

    # False Complaint Prevention
    cust_blob = " ".join(cust_lines).lower()
    if not any(k in cust_blob for k in ["frustrated", "angry", "again", "second time", "money", "work"]):
        if ov_sent == "Negative" or primary == "Complaint":
            logger.info("Downgrading False Complaint.")
            ov_sent, sent_conf, primary, conf = "Neutral", 0.55, "Inquiry", 0.50

    sent_r, intent_r = generate_reasoning(ov_sent, primary, cust_lines)
    
    return {
        "summary": {
            "overall_sentiment": ov_sent, "sentiment_confidence": round(sent_conf, 2), "sentiment_reason": sent_r,
            "primary_intent": primary, "intent_confidence": round(conf, 2), "intent_reason": intent_r,
            "empathy_score": emp_score, "empathy_confidence": emp_conf,
            "products_detected": sorted(detect_products(text, entities)), "entity_count": len(entities), "bert_enabled": True
        },
        "customer_queries": cust_lines, "agent_responses": agent_lines,
        "details": {"intents_found": unique_intents, "entities": entities}
    }

if __name__ == "__main__":
    path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "outputs", "analysis_result.json")
    with open(os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "sample_conversation.txt"), "r") as f:
        res = analyze_conversation(f.read())
    with open(path, "w") as f: json.dump(res, f, indent=4)
    print(f"✅ Saved to {path}")
