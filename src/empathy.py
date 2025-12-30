def detect_empathy(text):
    """
    Detects if the text contains empathetic language.
    """
    empathy_keywords = [
        "sorry", "apologize", "regret", "understand", "concern", 
        "appreciate", "patience", "unfortunate", "help you", 
        "assist you", "resolve", "worry"
    ]
    
    text_lower = text.lower()
    detected_keywords = [word for word in empathy_keywords if word in text_lower]
    
    return len(detected_keywords) > 0
