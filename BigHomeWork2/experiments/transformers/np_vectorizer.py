
import spacy
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.base import BaseEstimator, TransformerMixin

class NPVectorizer(BaseEstimator, TransformerMixin):
    def __init__(self, extra_features=[]):
        self.nlp = spacy.load("en_core_web_sm")
        self.vectorizer = CountVectorizer(ngram_range=(1, 1))
        self.extra_features = extra_features

    def fit(self, dataset, y=None):
        noun_phrases = []
        for text in dataset['text']:
            doc = self.nlp(text)
            phrases = [chunk.text for chunk in doc.noun_chunks]
            noun_phrases.append(" ".join(phrases))
        self.vectorizer.fit(noun_phrases)
        return self

    def transform(self, dataset):
        noun_phrases = []
        for text in dataset['text']:
            doc = self.nlp(text)
            phrases = [chunk.text for chunk in doc.noun_chunks]
            noun_phrases.append(" ".join(phrases))

        data = self.vectorizer.transform(noun_phrases)
        
        for feature in self.extra_features:
            data[feature] = dataset[feature]

        return data
        
