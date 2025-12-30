from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
import logging

logger = logging.getLogger(__name__)

# --- Training Data (Simulated for this standalone module) ---
TRAIN_DATA = [
    ("The product is broken and not working", "Complaint"),
    ("I am facing an issue with my order", "Complaint"),
    ("This is the worst service ever", "Complaint"),
    ("It arrived damaged", "Complaint"),
    ("Where is my order?", "Inquiry"),
    ("When will it be delivered?", "Inquiry"),
    ("How do I return this?", "Inquiry"),
    ("Can you help me with a refund?", "Support Request"),
    ("I need help resetting my password", "Support Request"),
    ("Thank you so much", "Feedback"),
    ("Great service, thanks", "Feedback"),
    ("You were very helpful", "Feedback"),
    ("I love this product", "Feedback")
]

model_pipeline = None

def train_intent_model():
    """
    Trains a simple TF-IDF + SVM intent classifier on usage.
    """
    global model_pipeline
    texts, labels = zip(*TRAIN_DATA)
    
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(ngram_range=(1, 2), stop_words='english')),
        ('clf', LinearSVC())
    ])
    
    logger.info("Training Intent Classifier...")
    pipeline.fit(texts, labels)
    model_pipeline = pipeline

# Train on import
train_intent_model()

def detect_intent(text):
    """
    Predicts intent using the trained TF-IDF + SVM model.
    """
    if not model_pipeline:
        return "Unknown"
        
    try:
        prediction = model_pipeline.predict([text])[0]
        return prediction
    except Exception as e:
        logger.error(f"Intent prediction failed: {e}")
        return "General"
