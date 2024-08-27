import spacy

def perform_ner(transcription):
    nlp = spacy.load("en_core_web_sm")
    doc = nlp(transcription)
    entities = [(entity.text, entity.label_) for entity in doc.ents]
    unique_entities = list(set(entities))

    print("\nExtracted Entities:")
    for entity in unique_entities:
        print(f"Entity: {entity[0]}, Label: {entity[1]}")
    return unique_entities
