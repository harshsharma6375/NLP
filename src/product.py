import spacy
from spacy.matcher import Matcher

# Load spaCy model (shared instance would be better in a class, but keeping functional per requirement)
try:
    nlp = spacy.load("en_core_web_sm")
except ImportError:
    import en_core_web_sm
    nlp = en_core_web_sm.load()

def detect_products(text, ner_entities=None):
    """
    Detects products using a combination of spaCy Rule-Based Matcher 
    and filtering specific NER types.
    """
    doc = nlp(text)
    matcher = Matcher(nlp.vocab)
    
    # 1. Define Patterns for common products
    # Pattern: "iPhone" followed by optional number
    pattern_iphone = [{"LOWER": "iphone"}, {"IS_DIGIT": True, "OP": "?"}]
    # Pattern: "Samsung" followed by optional "Galaxy" and model
    pattern_samsung = [{"LOWER": "samsung"}, {"LOWER": "galaxy", "OP": "?"}, {"SHAPE": "Xdd", "OP": "?"}]
    # Pattern: Keywords
    pattern_generic = [{"LOWER": {"IN": ["macbook", "ipad", "pixel", "kindle", "playstation", "xbox", "laptop", "monitor", "myntra", "flipkart", "amazon"]}}]

    matcher.add("TECH_PRODUCT", [pattern_iphone, pattern_samsung, pattern_generic])

    matches = matcher(doc)
    detected_products = set()

    for match_id, start, end in matches:
        span = doc[start:end]
        detected_products.add(span.text)

    # 2. Augment with NER entities if provided (filtering for Product-like ORGs or PRODUCTs)
    if ner_entities:
        for ent in ner_entities:
            if ent['label'] in ["PRODUCT", "ORG"]:
                # Simple heuristic: ignore common non-product ORGs if needed, 
                # but for now we include them as potential verified mentions.
                # In a real system, we'd have a whitelist/blacklist.
                pass 
                # We often find products in ORG (e.g., "Amazon" can be service, "Facebook" app)
                # For this specific assignment, let's trust the Matcher more for specific models 
                # and NER for general brands.
                detected_products.add(ent['text'])

    return list(detected_products)
