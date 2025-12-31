from textblob import TextBlob
import logging
from bert_manager import BertManager

logger = logging.getLogger(__name__)

def analyze_sentiment(text, use_bert=True):
    if use_bert:
        try:
            l, s = BertManager().predict_sentiment(text)
            return {"label": l, "score": round(s, 4), "polarity": 1.0 if l == "POSITIVE" else -1.0, "confidence": min(round(s, 2), 0.90), "source": "distilbert"}
        except Exception: pass

    blob = TextBlob(text)
    p = blob.sentiment.polarity
    label = "POSITIVE" if p > 0.1 else "NEGATIVE" if p < -0.1 else "NEUTRAL"
    return {"label": label, "score": round(abs(p), 4), "polarity": round(p, 4), "confidence": round(min(abs(p) * 2, 1.0), 4), "source": "textblob"}
