import logging

logger = logging.getLogger(__name__)

def detect_empathy(text):
    """
    Detects empathy using robust keyword matching.
    Focuses on Apology, Understanding, and Helpfulness.
    """
    text_lower = text.lower()
    
    # Categories of empathetic language
    apology_words = ["sorry", "apologize", "regret", "pardon", "forgive"]
    understanding_words = ["understand", "realize", "hear you", "frustration", "concern", "worry"]
    action_words = ["help", "assist", "resolve", "fix", "right away", "immediately"]
    politeness_words = ["please", "thank you", "appreciate", "patience"]

    score = 0
    detected_categories = []

    if any(w in text_lower for w in apology_words):
        score += 0.4
        detected_categories.append("apology")
        
    if any(w in text_lower for w in understanding_words):
        score += 0.3
        detected_categories.append("understanding")
        
    if any(w in text_lower for w in action_words):
        score += 0.2
        detected_categories.append("action")
        
    if any(w in text_lower for w in politeness_words):
        score += 0.1
        detected_categories.append("politeness")

    # Normalize score cap at 1.0
    final_score = min(score, 1.0)
    
    return {
        "detected": final_score > 0,
        "primary_types": detected_categories,
        "confidence": round(final_score, 2)
    }
