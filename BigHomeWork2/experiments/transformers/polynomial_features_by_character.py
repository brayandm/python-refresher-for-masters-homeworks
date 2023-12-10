
from sklearn.preprocessing import PolynomialFeatures
from sklearn.base import BaseEstimator, TransformerMixin

class PolynomialFeaturesByCharacter(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.characters = [
            "ironman", "captainamerica", "hulk", "thor", "blackwidow", "hawkeye",
            "thanos", "antman", "captainmarvel", "spiderman", "doctorstrange",
            "blackpanther", "nebula", "gamora", "loki", "scarletwitch", "vision",
            "falcon", "wintersoldier", "starlord", "drax", "groot", "rocket",
            "mantis", "okoye", "shuri", "pepperpotts", "happyhogan", "nickfury",
            "mariahill", "wong", "hankpym", "janetvandyne", "wasp"
        ]
        self.poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)

    def fit(self, dataset, y=None):
        self.poly.fit(dataset[self.characters])
        return self

    def transform(self, dataset):
        return self.poly.transform(dataset[self.characters])
