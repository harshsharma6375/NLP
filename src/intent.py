import logging
def detect_intent(text):
    t = text.lower()
    
    # Keywords
    positive_words = ["happy", "great", "improved", "satisfied", "love"]
    negative_words = ["bad", "broken", "frustrated", "frustrating", "issue", "slow", "drains", "dies"]
    
    implicit_patterns = [
        "only", "lasts", "after charging", "less than",
        "takes too long", "not working", "keeps crashing",
        "one hour", "two hours"
    ]
    
    failure_phrases = [
        "dies", "drains", "not working", "keeps crashing",
        "one hour", "two hours"
    ]

    pos_hits = [w for w in positive_words if w in t]
    neg_hits = [w for w in negative_words if w in t]
    implicit_issue = any(p in t for p in implicit_patterns)
    sarcasm = bool(pos_hits and any(p in t for p in failure_phrases))

    if neg_hits or implicit_issue or sarcasm:
        return "Complaint", pos_hits, neg_hits, implicit_issue, sarcasm
    elif pos_hits:
        return "Feedback", pos_hits, neg_hits, implicit_issue, sarcasm
    else:
        return "Inquiry", pos_hits, neg_hits, implicit_issue, sarcasm
