
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.base import BaseEstimator, TransformerMixin

class WordVectorizer2GramByCharacter(BaseEstimator, TransformerMixin):
    def __init__(self, extra_features=[]):
        self.vectorizer = CountVectorizer(analyzer='char', ngram_range=(2, 2))
        self.extra_features = extra_features
    
    def fit(self, dataset, y=None):
        self.vectorizer.fit(dataset['screenName'])
        return self

    def transform(self, dataset):
        data =  self.vectorizer.transform(dataset['screenName'])
        
        for feature in self.extra_features:
            data[feature] = dataset[feature]

        return data
