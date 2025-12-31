def detect_products(text, entities):
    return [e for e in entities if "samsung" in e.lower()]
