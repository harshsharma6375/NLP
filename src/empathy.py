import logging
from bert_manager import BertManager
logger = logging.getLogger(__name__)

def detect_empathy(text, use_bert=True):
    text_l = text.lower()
    signals = sum(1 for w in ["sorry", "apologize", "understand", "realize", "help", "resolve"] if w in text_l)
    
    h_score = 0.85 if signals >= 3 else (0.6 if signals >= 1 else 0.0)
    
    if use_bert:
        try:
            _, b_conf, label = BertManager().predict_empathy(text)
            b_score = 0.3 if label in ['joy', 'neutral'] else (0.6 if label == 'sadness' else 0.0)
            final_score = max(b_score, h_score)
            if label in ['anger', 'disgust']: final_score = 0.1
            
            return {"detected": final_score > 0.1, "primary_emotion": label, "confidence": min(round(0.75 + (b_conf * 0.15), 2), 0.90), "score": round(final_score, 2), "source": "bert"}
        except Exception: pass

    return {"detected": h_score > 0, "primary_emotion": "heuristic", "confidence": 0.85 if h_score > 0 else 0.0, "score": round(h_score, 2), "source": "heuristic"}
