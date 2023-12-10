
from sklearn.base import BaseEstimator, ClassifierMixin

from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

import experiments.transformers.np_vectorizer

class ExpHW6Classifier2(BaseEstimator, ClassifierMixin):

    def __init__(self):
        self.pipeline = Pipeline([
            ('np_vectorizer', experiments.transformers.np_vectorizer.NPVectorizer()),
            ('truncated_svd', TruncatedSVD(n_components=2)),
            ('logistic_regression', LogisticRegression())
        ])

    def fit(self, dataset, y=None):
        self.pipeline.fit(dataset, y)
        return self

    def predict(self, dataset):
        return self.pipeline.predict(dataset)
