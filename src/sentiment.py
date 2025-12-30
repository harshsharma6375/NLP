from textblob import TextBlob
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def analyze_sentiment(text):
    """
    Analyzes sentiment using TextBlob (Lexicon-based).
    Fast and requires no model loading.
    Returns dictionary with label and score/polarity.
    """
    blob = TextBlob(text)
    polarity = blob.sentiment.polarity
    
    label = "NEUTRAL"
    if polarity > 0.1:
        label = "POSITIVE"
    elif polarity < -0.1:
        label = "NEGATIVE"

    return {
        "source": "textblob",
        "label": label,
        "score": round(abs(polarity), 4),
        "polarity": round(polarity, 4)
    }
