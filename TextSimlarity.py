import numpy as np
import Morpheme
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from LSA import Bag_of_Words, SVD
class TextSimilarity:
    def __init__(self):
        self.vectorize = TfidfVectorizer()
        self.tfidf_ = None
        self.bow = Bag_of_Words()
        self.svd = SVD()

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
    
    def fit_compression_transform(self, corpus:list):
        bow_data = self.bow.fit_transform(corpus).toarray()
        svd_data = self.svd.fit_transform(bow_data)
        return cosine_similarity(svd_data)
        