import logging
from transformers import pipeline

logger = logging.getLogger(__name__)

class BertManager:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(BertManager, cls).__new__(cls)
            cls._instance.sentiment_pipe = None
            cls._instance.emotion_pipe = None
            cls._instance.intent_pipe = None
        return cls._instance

    def _load(self, type_):
        # Strict "Low Temperature" equivalent: top_k=1 ensures deterministic greedy decoding
        if type_ == 'sentiment' and not self.sentiment_pipe:
            logger.info("Loading DistilBERT (Deterministic)...")
            self.sentiment_pipe = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english", top_k=1)
        elif type_ == 'emotion' and not self.emotion_pipe:
            logger.info("Loading RoBERTa (Deterministic)...")
            self.emotion_pipe = pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base", top_k=1)
        elif type_ == 'intent' and not self.intent_pipe:
            logger.info("Loading BART (Deterministic)...")
            self.intent_pipe = pipeline("zero-shot-classification", model="facebook/bart-large-mnli", top_k=1)

    def predict_sentiment(self, text):
        self._load('sentiment')
        res = self.sentiment_pipe(text[:512])[0] # Deterministic output
        return res['label'], res['score']

    def predict_empathy(self, text):
        self._load('emotion')
        res = self.emotion_pipe(text[:512])[0]
        return res['label'] in ['joy', 'neutral', 'surprise'], res['score'], res['label']

    def predict_intent(self, text, candidates):
        self._load('intent')
        res = self.intent_pipe(text[:512], candidates)
        return res['labels'][0], res['scores'][0]
