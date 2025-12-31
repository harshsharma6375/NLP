import logging

logger = logging.getLogger(__name__)

def detect_empathy(text):
    t = text.lower()
    strong = ["sorry", "apologize", "understand"]
    ack = ["thank", "appreciate", "glad", "great to hear"]

    if any(w in t for w in strong):
        return "strong"
    if any(w in t for w in ack):
        return "ack"
    return "none"
