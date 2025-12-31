import logging
from bert_manager import BertManager

INTENT_LABELS = ["Complaint", "Delivery Delay", "Refund Issue", "Payment Issue", "Inquiry", "Feedback", "Service Dissatisfaction"]
logger = logging.getLogger(__name__)

def detect_intent(text, use_bert=True):
    if use_bert:
        try:
            l, s = BertManager().predict_intent(text, INTENT_LABELS)
            return {"label": l, "score": round(s, 4), "source": "bert"}
        except Exception: pass

    text = text.lower()
    label, score = "Inquiry", 0.5
    
    if "refund" in text: label, score = "Refund Issue", 0.65
    elif any(x in text for x in ["late", "deliver", "arrive"]): label, score = "Delivery Delay", 0.65
    elif any(x in text for x in ["monitor", "price", "cost"]): label, score = "Inquiry", 0.5
    elif any(x in text for x in ["frustrat", "disappoint", "angry"]): label, score = "Complaint", 0.8

    return {"label": label, "score": score, "source": "keyword"}
