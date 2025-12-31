import json
import os
import uuid
import logging
from ner import extract_entities
from product import detect_products
from sentiment import analyze_sentiment
from intent import detect_intent
from empathy import detect_empathy

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

# =======================
# SESSION STATE
# =======================

class Session:
    def __init__(self):
        self.session_id = str(uuid.uuid4())
        self.history = []
        self.cache = {
            "pos_words": [],
            "neg_words": [],
            "implicit_issue": False,
            "sarcasm": False,
            "intents": set(),
            "entities": set(),
            "products": set(),
            "empathy_ack": 0,
            "empathy_strong": 0
        }

    def update_customer(self, analysis):
        self.cache["pos_words"].extend(analysis["pos_words"])
        self.cache["neg_words"].extend(analysis["neg_words"])
        self.cache["implicit_issue"] |= analysis["implicit_issue"]
        self.cache["sarcasm"] |= analysis["sarcasm"]
        self.cache["intents"].add(analysis["intent"])
        self.cache["entities"].update(analysis["entities"])
        self.cache["products"].update(analysis["products"])

    def update_agent(self, empathy):
        if empathy == "strong":
            self.cache["empathy_strong"] += 1
        elif empathy == "ack":
            self.cache["empathy_ack"] += 1

# =======================
# FINAL OUTPUT
# =======================

def build_output(session):
    c = session.cache

    # ---------- FINAL INTENT CLEANUP ----------
    final_intents = sorted(list(c["intents"]))
    
    if "Complaint" in c["intents"]:
        primary_intent = "Complaint"
    elif "Feedback" in c["intents"]:
        primary_intent = "Feedback"
    else:
        primary_intent = "Inquiry"

    # ---------- SENTIMENT (ALIGNED WITH INTENT) ----------
    if primary_intent == "Complaint":
        overall_sentiment = "Negative"
        sentiment_confidence = 0.9
        sentiment_reason = "Customer reports a problem or dissatisfaction with the product."
    elif primary_intent == "Feedback":
        overall_sentiment = "Positive"
        sentiment_confidence = 0.9
        sentiment_reason = (
            "Customer expresses happiness using words like "
            + ", ".join(set(c["pos_words"])) + "."
        )
    else:
        overall_sentiment = "Neutral"
        sentiment_confidence = 0.6
        sentiment_reason = "No strong emotional expressions detected."

    # ---------- EMPATHY ----------
    if c["empathy_strong"] > 0:
        empathy_score = 1.0
    elif c["empathy_ack"] > 0:
        empathy_score = 0.5
    else:
        empathy_score = 0.0

    return {
        "session_id": session.session_id,
        "summary": {
            "overall_sentiment": overall_sentiment,
            "sentiment_confidence": sentiment_confidence,
            "sentiment_reason": sentiment_reason,
            "primary_intent": primary_intent,
            "intent_confidence": 0.8 if primary_intent != "Inquiry" else 0.6,
            "intent_reason": f"Customer interaction indicates {primary_intent}.",
            "empathy_score": empathy_score,
            "empathy_confidence": 0.85,
            "products_detected": sorted(list(c["products"])),
            "entity_count": len(c["entities"]),
            "bert_enabled": True 
        },
        "details": {
            "intents_found": list(final_intents)
        }
    }

# =======================
# PIPELINE
# =======================

def analyze_conversation(text):
    session = Session()

    for line in text.split("\n"):
        if ":" not in line:
            continue

        role, msg = line.split(":", 1)
        role = role.lower().strip()
        msg = msg.strip()

        session.history.append({"role": role, "text": msg})

        if "customer" in role:
            # Call modular functions
            entities = extract_entities(msg)
            products = detect_products(msg, entities)
            # Intent function returns tuple: (intent, pos, neg, implicit, sarcasm)
            intent, pos_h, neg_h, imp, sarc = detect_intent(msg)
            
            analysis = {
                "intent": intent,
                "pos_words": pos_h,
                "neg_words": neg_h,
                "implicit_issue": imp,
                "sarcasm": sarc,
                "entities": entities,
                "products": products
            }
            session.update_customer(analysis)

        elif "agent" in role:
            empathy = detect_empathy(msg)
            session.update_agent(empathy)

    output = build_output(session)
    output["customer_queries"] = [h["text"] for h in session.history if "customer" in h["role"]]
    output["agent_responses"] = [h["text"] for h in session.history if "agent" in h["role"]]

    return output

# =======================
# RUN
# =======================

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    input_path = os.path.join(base_dir, "data", "sample_conversation.txt")
    output_dir = os.path.join(base_dir, "outputs")
    output_path = os.path.join(output_dir, "analysis_result.json")

    os.makedirs(output_dir, exist_ok=True)

    print("📥 Reading from:", input_path)
    print("📤 Writing to :", output_path)

    with open(input_path, "r", encoding="utf-8") as f:
        text = f.read()

    result = analyze_conversation(text)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=4, ensure_ascii=False)
        f.flush()
        os.fsync(f.fileno())

    print("✅ Output written successfully")
