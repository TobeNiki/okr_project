import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class TextSimilarity:
    def __init__(self):
        self.vectorize = TfidfVectorizer()
        self.tfidf_ = None

    def _tfidf(self, corpus:list):
        tfidf_x = self.vectorize.fit_transform(corpus)
        self.tfidf_ = tfidf_x.toarray()

    def _cossimilarity(self):
        return cosine_similarity(self.tfidf_)

    def get_feature_name(self)->list:
        return self.vectorize.get_feature_names()

    def get_tfidf(self):
        return self.tfidf_

    def fit_transform(self, corpus:list):
        self._tfidf(corpus=corpus)
        return self._cossimilarity()