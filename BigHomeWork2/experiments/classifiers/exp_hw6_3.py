
from sklearn.base import BaseEstimator, ClassifierMixin

from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

import experiments.transformers.word_vectorizer_2gram

class ExpHW6Classifier3(BaseEstimator, ClassifierMixin):

    def __init__(self):
        self.pipeline = Pipeline([
            ('word_vectorizer_2gram', experiments.transformers.word_vectorizer_2gram.WordVectorizer2Gram()),
            ('truncated_svd', TruncatedSVD(n_components=2)),
            ('logistic_regression', LogisticRegression())
        ])

    def fit(self, dataset, y=None):
        self.pipeline.fit(dataset, y)
        return self

    def predict(self, dataset):
        return self.pipeline.predict(dataset)
