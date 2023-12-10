
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.base import BaseEstimator, TransformerMixin
from scipy import sparse

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
            if dataset[feature].dtype == 'bool':
                extra_data = dataset[feature].astype(int)
            else:
                extra_data = dataset[feature]

            extra_data_sparse = sparse.csr_matrix(extra_data).T 
            data = sparse.hstack([data, extra_data_sparse], format='csr')

        return data
