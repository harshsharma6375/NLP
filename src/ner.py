def extract_entities(text):
    entities = []
    known_entities = [
        "Samsung Galaxy",
        "battery",
        "battery life",
        "camera",
        "performance",
        "update"
    ]
    for e in known_entities:
        if e.lower() in text.lower():
            entities.append(e)
    return entities
