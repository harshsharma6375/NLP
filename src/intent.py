def detect_intent(text):
    text = text.lower()
    
    intents = {
        "greeting": ["hello", "hi ", "hey", "good morning", "good evening"],
        "closing": ["bye", "goodbye", "see you", "have a nice day"],
        "complaint": ["problem", "issue", "error", "broken", "fail", "not working", "bad", "worst"],
        "query": ["what", "how", "where", "when", "why", "?"],
        "purchase": ["buy", "order", "purchase", "cost", "price"],
        "feedback": ["thanks", "thank you", "good job", "great", "excellent", "love"]
    }

    detected_intents = []
    
    for intent, keywords in intents.items():
        for word in keywords:
            if word in text:
                detected_intents.append(intent)
                break  # match one keyword per intent is enough
    
    if not detected_intents:
        return "general_statement"
    
    # Return the most significant found (or all of them joined)
    # For simplicity, returning the first distinct one that isn't greeting if multiple exist, 
    # or just comma separated.
    
    # Priority: Complaint > Purchase > Query > Feedback > Greeting/Closing
    priority_order = ["complaint", "purchase", "query", "feedback", "greeting", "closing"]
    
    for p in priority_order:
        if p in detected_intents:
            return p
            
    return detected_intents[0]
