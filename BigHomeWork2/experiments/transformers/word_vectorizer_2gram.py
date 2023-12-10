
import spacy
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.base import BaseEstimator, TransformerMixin

class WordVectorizer2Gram(BaseEstimator, TransformerMixin):
    def __init__(self, extra_features=[]):
        self.nlp = spacy.load("en_core_web_sm")
        self.vectorizer = CountVectorizer(tokenizer=self.custom_tokenizer, ngram_range=(2, 2))
        self.extra_features = extra_features

    def custom_tokenizer(self, text):
        doc = self.nlp(text)
        nnps = [token.text for token in doc if token.pos_ == 'PROPN']
        return nnps

    def fit(self, dataset, y=None):
        self.vectorizer.fit(dataset['text'])
        return self

    def transform(self, dataset):
        data =  self.vectorizer.transform(dataset['text'])

        for feature in self.extra_features:
            data[feature] = dataset[feature]

        return data
