
from sklearn.base import BaseEstimator, ClassifierMixin

from sklearn.decomposition import TruncatedSVD
from xgboost import XGBClassifier
from sklearn.pipeline import Pipeline

from importlib import reload
import experiments.transformers.np_vectorizer
reload(experiments.transformers.np_vectorizer)

from sklearn.preprocessing import FunctionTransformer

class ExpHW6Classifier21(BaseEstimator, ClassifierMixin):

    def __init__(self, n_components=2, extra_features=[], n_estimators=100, learning_rate=0.05):
        self.transformer = Pipeline([
            ('np_vectorizer', experiments.transformers.np_vectorizer.NPVectorizer()),
            ('truncated_svd', TruncatedSVD(n_components=n_components)),
        ])

        def append_extra_features(X, extra_features):
            return np.hstack((X, extra_features)) if extra_features is not None else X

        self.classifier = Pipeline([
            ('feature_combiner', FunctionTransformer(append_extra_features, validate=False)),
            ('xgb_classifier', XGBClassifier(
                n_estimators=n_estimators,
                learning_rate=learning_rate
            ))
        ])

    def fit(self, dataset, y=None, extra_features=None):
        X_transformed = self.transformer.fit_transform(dataset)
        self.classifier.set_params(feature_combiner__kw_args={'extra_features': extra_features})
        self.classifier.fit(X_transformed, y)
        return self

    def predict(self, dataset, extra_features=None):
        X_transformed = self.transformer.transform(dataset)
        self.classifier.set_params(feature_combiner__kw_args={'extra_features': extra_features})
        return self.classifier.predict(X_transformed)
