from keybert import KeyBERT
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

def lemmatize_text(text):
    lemmatizer = WordNetLemmatizer()
    tokens = word_tokenize(text)
    lemmatized_text = ' '.join([lemmatizer.lemmatize(token) for token in tokens])
    return lemmatized_text

def extract_keywords(transcription, num_keywords=10):
    model = KeyBERT()
    lemmatized_transcription = lemmatize_text(transcription)
    keywords = model.extract_keywords(lemmatized_transcription, keyphrase_ngram_range=(1, 2), top_n=num_keywords)
    print("\nKeywords:\n", keywords)
    return keywords
