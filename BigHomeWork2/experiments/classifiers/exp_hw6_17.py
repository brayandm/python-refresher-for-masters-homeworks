
from sklearn.base import BaseEstimator, ClassifierMixin

from sklearn.decomposition import TruncatedSVD
from xgboost import XGBClassifier
from sklearn.pipeline import Pipeline

from importlib import reload
import experiments.transformers.word_vectorizer_1gram
reload(experiments.transformers.word_vectorizer_1gram)

import experiments.transformers.np_vectorizer
reload(experiments.transformers.np_vectorizer)

import experiments.transformers.word_vectorizer_2gram
reload(experiments.transformers.word_vectorizer_2gram)

import experiments.transformers.word_vectorizer_2gram_by_character
reload(experiments.transformers.word_vectorizer_2gram_by_character)

import experiments.transformers.polynomial_features_by_character
reload(experiments.transformers.polynomial_features_by_character)

from sklearn.preprocessing import FunctionTransformer

import numpy as np

class ExpHW6Classifier17(BaseEstimator, ClassifierMixin):

    def __init__(self, n_components=2, extra_features=[], n_estimators=100, learning_rate=0.05):

        self.transformer = {}

        self.transformer[0] = Pipeline([
            ('word_vectorizer_1gram', experiments.transformers.word_vectorizer_1gram.WordVectorizer1Gram()),
            ('truncated_svd', TruncatedSVD(n_components=10)),
        ])

        self.transformer[1] = Pipeline([
            ('np_vectorizer', experiments.transformers.np_vectorizer.NPVectorizer()),
            ('truncated_svd', TruncatedSVD(n_components=10)),
        ])

        self.transformer[2] = Pipeline([
            ('word_vectorizer_2gram', experiments.transformers.word_vectorizer_2gram.WordVectorizer2Gram()),
            ('truncated_svd', TruncatedSVD(n_components=10)),
        ])

        self.transformer[3] = Pipeline([
            ('word_vectorizer_2gram_by_character', experiments.transformers.word_vectorizer_2gram_by_character.WordVectorizer2GramByCharacter()),
            ('truncated_svd', TruncatedSVD(n_components=10)),
        ])

        self.transformer[4] = Pipeline([
            ('polynomial_features_by_character', experiments.transformers.polynomial_features_by_character.PolynomialFeaturesByCharacter()),
            ('truncated_svd', TruncatedSVD(n_components=10)),
        ])

        self.classifier = XGBClassifier(
            n_estimators=n_estimators,
            learning_rate=learning_rate
        )

    def fit(self, dataset, y=None, extra_features=None):
        concated_results = None

        for i in range(5):

            result = self.transformer[i].fit_transform(dataset)

            if concated_results is None:
                concated_results = result
            else:
                concated_results = np.hstack((concated_results, result))
                
        self.classifier.fit(concated_results, dataset['Target'])

        return self

    def predict(self, dataset, extra_features=None):
        concated_results = None

        for i in range(5):

            result = self.transformer[i].fit_transform(dataset)

            if concated_results is None:
                concated_results = result
            else:
                concated_results = np.hstack((concated_results, result))

        return self.classifier.predict(concated_results)
