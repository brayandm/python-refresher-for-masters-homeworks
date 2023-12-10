
from sklearn.preprocessing import PolynomialFeatures
from sklearn.base import BaseEstimator, TransformerMixin
from scipy import sparse

class PolynomialFeaturesByCharacter(BaseEstimator, TransformerMixin):
    def __init__(self, extra_features=[]):
        self.characters = [
            "ironman", "captainamerica", "hulk", "thor", "blackwidow", "hawkeye",
            "thanos", "antman", "captainmarvel", "spiderman", "doctorstrange",
            "blackpanther", "nebula", "gamora", "loki", "scarletwitch", "vision",
            "falcon", "wintersoldier", "starlord", "drax", "groot", "rocket",
            "mantis", "okoye", "shuri", "pepperpotts", "happyhogan", "nickfury",
            "mariahill", "wong", "hankpym", "janetvandyne", "wasp"
        ]
        self.poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
        self.extra_features = extra_features

    def fit(self, dataset, y=None):
        self.poly.fit(dataset[self.characters])
        return self

    def transform(self, dataset):
        data = self.poly.transform(dataset[self.characters])

        for feature in self.extra_features:
            if dataset[feature].dtype == 'bool':
                extra_data = dataset[feature].astype(int)
            else:
                extra_data = dataset[feature]

            extra_data_sparse = sparse.csr_matrix(extra_data).T 
            data = sparse.hstack([data, extra_data_sparse], format='csr')

        return data
