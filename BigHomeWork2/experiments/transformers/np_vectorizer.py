
import spacy
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.base import BaseEstimator, TransformerMixin
from scipy import sparse

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
            if dataset[feature].dtype == 'bool':
                extra_data = dataset[feature].astype(int)
            else:
                extra_data = dataset[feature]

            extra_data_sparse = sparse.csr_matrix(extra_data).T 
            data = sparse.hstack([data, extra_data_sparse], format='csr')

        return data
        
