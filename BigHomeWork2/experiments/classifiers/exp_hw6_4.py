
from sklearn.base import BaseEstimator, ClassifierMixin

from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from importlib import reload
import experiments.transformers.word_vectorizer_2gram_by_character
reload(experiments.transformers.word_vectorizer_2gram_by_character)

class ExpHW6Classifier4(BaseEstimator, ClassifierMixin):

    def __init__(self, n_components=2, extra_features=[]):
        self.pipeline = Pipeline([
            ('word_vectorizer_2gram_by_character', experiments.transformers.word_vectorizer_2gram_by_character.WordVectorizer2GramByCharacter()),
            ('truncated_svd', TruncatedSVD(n_components=n_components)),
            ('logistic_regression', LogisticRegression())
        ])

    def fit(self, dataset, y=None):
        self.pipeline.fit(dataset, y)
        return self

    def predict(self, dataset):
        return self.pipeline.predict(dataset)
