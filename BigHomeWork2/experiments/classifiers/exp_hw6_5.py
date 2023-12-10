
from sklearn.base import BaseEstimator, ClassifierMixin

from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

import experiments.transformers.polynomial_features_by_character

class ExpHW6Classifier5(BaseEstimator, ClassifierMixin):

    def __init__(self, n_components=2):
        self.pipeline = Pipeline([
            ('polynomial_features_by_character', experiments.transformers.polynomial_features_by_character.PolynomialFeaturesByCharacter()),
            ('truncated_svd', TruncatedSVD(n_components=n_components)),
            ('logistic_regression', LogisticRegression())
        ])

    def fit(self, dataset, y=None):
        self.pipeline.fit(dataset, y)
        return self

    def predict(self, dataset):
        return self.pipeline.predict(dataset)
