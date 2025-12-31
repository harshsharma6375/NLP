import spacy
from spacy.matcher import Matcher

try: nlp = spacy.load("en_core_web_sm")
except ImportError: import en_core_web_sm; nlp = en_core_web_sm.load()

def detect_products(text, ner_entities=None):
    doc = nlp(text)
    matcher = Matcher(nlp.vocab)
    
    patterns = [
        [{"LOWER": "iphone"}, {"IS_DIGIT": True, "OP": "?"}, {"LOWER": "pro", "OP": "?"}, {"LOWER": "max", "OP": "?"}],
        [{"LOWER": "samsung"}, {"LOWER": "galaxy", "OP": "?"}, {"SHAPE": "Xdd", "OP": "?"}],
        [{"LOWER": "lenovo"}, {"LOWER": "laptop", "OP": "?"}],
        [{"LOWER": "amazon", "OP": "?"}, {"LOWER": "echo"}, {"LOWER": "dot", "OP": "?"}, {"LOWER": "show", "OP": "?"}],
        [{"LOWER": "fire"}, {"LOWER": "tv"}],
        [{"LOWER": "kindle"}, {"LOWER": "paperwhite", "OP": "?"}],
        [{"LOWER": "microsoft", "OP": "?"}, {"LOWER": "surface"}, {"LOWER": "pro", "OP": "?"}, {"LOWER": "laptop", "OP": "?"}]
    ]
    matcher.add("TECH_PRODUCT", patterns)
    
    detected = {doc[start:end].text for _, start, end in matcher(doc)}
    
    if ner_entities:
        for ent in ner_entities:
            if ent['label'] == "PRODUCT": detected.add(ent['text'])

    final_products = []
    casing = {"iphone": "iPhone", "ipad": "iPad", "ios": "iOS", "macos": "macOS"}
    
    for p in sorted(detected, key=len, reverse=True):
        p_clean = p.title()
        for k, v in casing.items():
            if k in p_clean.lower(): p_clean = p_clean.replace(k.title(), v).replace(k.lower(), v)
        if not any(p_clean in exist for exist in final_products): final_products.append(p_clean)
            
    return final_products
