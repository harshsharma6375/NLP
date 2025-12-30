import spacy

# Load spaCy model once
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    from spacy.cli import download
    download("en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")

def extract_entities(text):
    """
    Extracts named entities from text using spaCy.
    Filters for relevant types: ORG, GPE, PERSON, PRODUCT, DATE, MONEY.
    """
    doc = nlp(text)
    entities = []
    
    # Define relevant entity labels for this domain
    RELEVANT_LABELS = {"ORG", "GPE", "PERSON", "PRODUCT", "DATE", "MONEY"}

    for ent in doc.ents:
        if ent.label_ in RELEVANT_LABELS:
            entities.append({
                "text": ent.text,
                "label": ent.label_
            })

    # Return unique entities to avoid clutter (deduplication by text+label)
    # Using a dictionary comprehension to dedup
    unique_entities = {f"{e['text']}_{e['label']}": e for e in entities}.values()
    
    return list(unique_entities)
