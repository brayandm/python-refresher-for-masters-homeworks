
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.base import BaseEstimator, TransformerMixin

class WordVectorizer1Gram(BaseEstimator, TransformerMixin):
    
    def __init__(self):
        self.vectorizer = CountVectorizer(ngram_range=(1, 1))
    
    def fit(self, dataset, y=None):
        self.vectorizer.fit(dataset['text'])
        return self
    
    def transform(self, dataset):
        return self.vectorizer.transform(dataset['text'])
