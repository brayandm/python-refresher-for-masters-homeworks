
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.base import BaseEstimator, TransformerMixin

class WordVectorizer1Gram(BaseEstimator, TransformerMixin):
    
    def __init__(self, extra_features=[]):
        self.vectorizer = CountVectorizer(ngram_range=(1, 1))
        self.extra_features = extra_features
    
    def fit(self, dataset, y=None):
        self.vectorizer.fit(dataset['text'])
        return self
    
    def transform(self, dataset):
        data = self.vectorizer.transform(dataset['text'])

        for feature in self.extra_features:
            data[feature] = dataset[feature]

        return data
