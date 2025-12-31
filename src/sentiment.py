from textblob import TextBlob
import logging
from bert_manager import BertManager

logger = logging.getLogger(__name__)

def analyze_sentiment(text):
    t = text.lower()
    positive_words = ["happy", "great", "improved", "satisfied", "love"]
    negative_words = ["bad", "broken", "frustrated", "frustrating", "issue", "slow", "drains", "dies"]

    if any(w in t for w in negative_words):
        return "Negative"
    if any(w in t for w in positive_words):
        return "Positive"
    return "Neutral"
