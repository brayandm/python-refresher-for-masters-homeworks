
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.base import BaseEstimator, TransformerMixin

class WordVectorizer2GramByCharacter(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.vectorizer = CountVectorizer(analyzer='char', ngram_range=(2, 2))
    
    def fit(self, dataset, y=None):
        self.vectorizer.fit(dataset['screenName'])
        return self

    def transform(self, dataset):
        return self.vectorizer.transform(dataset['screenName'])
