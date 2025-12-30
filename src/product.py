def detect_products(text, entities):
    """
    Detects products based on named entities and keyword matching.
    """
    products = []
    
    # Keyword based detection (common tech products)
    product_keywords = [
        "iphone", "ipad", "macbook", "galaxy", "samsung", "pixel", 
        "laptop", "phone", "monitor", "keyboard", "mouse", "headset"
    ]
    
    text_lower = text.lower()
    for keyword in product_keywords:
        if keyword in text_lower:
            # Check if not already added to avoid duplicates
            if keyword not in products:
                products.append(keyword)

    # Entity based detection (filtering for products/orgs that might be products)
    for ent in entities:
        # SpaCy labels: PRODUCT (objects, vehicles, foods, etc.), ORG (companies, agencies, etc.)
        if ent['label'] in ["PRODUCT", "ORG"]:
            # Avoid duplicates and check if it looks like a product (simple heuristic)
            if ent['text'].lower() not in products:
                 products.append(ent['text'])
                 
    return products
