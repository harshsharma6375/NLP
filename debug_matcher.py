import spacy
from spacy.matcher import Matcher

nlp = spacy.load("en_core_web_sm")
doc = nlp("I received my order from Myntra earlier than expected.")

matcher = Matcher(nlp.vocab)
pattern_generic = [{"LOWER": {"IN": ["macbook", "ipad", "pixel", "kindle", "playstation", "xbox", "laptop", "monitor", "myntra", "flipkart", "amazon"]}}]
matcher.add("TECH_PRODUCT", [pattern_generic])

matches = matcher(doc)
print(f"Matches found: {len(matches)}")
for match_id, start, end in matches:
    print(f"Match: {doc[start:end].text}")
