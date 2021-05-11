from sklearn.feature_extraction.text import CountVectorizer

class Bag_of_Words:
    def __init__(self):
        self.vectorizer = CountVectorizer()

    def fit_transform(self, texts:list):
        self.vectorizer.fit(texts)
        return self.vectorizer.transform(texts)
    
    def get_feature_name(self):
        return self.vectorizer.get_feature_names()

from sklearn.decomposition import TruncatedSVD

class SVD:
    def __init__(self, n_components:int=4, random_state:int=42):
        self.svd = TruncatedSVD(n_components=n_components, random_state=random_state)
                
    def fit_transform(self, bow):
        self.svd.fit(bow)
        return self.svd.transform(bow)

    def get_n_components(self):
        return self.svd.n_components


