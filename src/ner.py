import spacy

try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    from spacy.cli import download
    download("en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")

def extract_entities(text):
    doc = nlp(text)
    labels = {"ORG", "GPE", "PERSON", "PRODUCT", "DATE", "MONEY"}
    entities = {} # dedup
    for ent in doc.ents:
        if ent.label_ in labels:
            entities[f"{ent.text}_{ent.label_}"] = {"text": ent.text, "label": ent.label_}
    return list(entities.values())
