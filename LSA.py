from sklearn.feature_extraction.text import CountVectorizer
from Morpheme import Morpheme
import numpy as np
from draw import draw_barcharts

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
    def __init__(self, n_components=4, random_state=42):
        self.svd = TruncatedSVD(n_components=n_components, random_state=random_state)
                
    def fit_transform(self, bow):
        self.svd.fit(bow)
        return self.svd.transform(bow)

    def get_n_components(self):
        return self.svd.n_components

    def fit_transform_of_np(self,bow):
        return np.linalg.svd(bow)


if __name__ == "__main__":
    bow = Bag_of_Words()
    text = [
        'このコードではMeCab.Taggerクラスのインスタンスを生成。',
        'そのparseメソッドに「くるまでまつ」という文字列を渡すことで解析を実行しています。',
        'その実行結果は次のようになりました。',
        'MeCab.Taggerクラスのインスタンス生成を行っている行にはコメントがあります。',
        '形態素解析実行時に参照する辞書を変更するといったことが可能です。'
    ]
    mor = Morpheme()
    morpheme = mor.fit_transform(text)
    print(morpheme)
    bow_value = bow.fit_transform(morpheme)
    print(bow_value.toarray())
    import pandas as pd
    import matplotlib.pyplot as plt
    import japanize_matplotlib
    bow_table = pd.DataFrame(bow_value.toarray(), columns=bow.get_feature_name())
    print(bow_table)
    #draw_barcharts(bow_value.toarray(), bow.get_feature_name(), text)
    svd = SVD()
    decom = svd.fit_transform(bow_table)
    print(decom)
    from TextSimlarity import TextSimilarity
    ts1 = TextSimilarity()
    print(ts1.fit_transform(morpheme))
    ts = TextSimilarity()
    ts.tfidf_ = decom
    cos = ts._cossimilarity()
    #draw_barcharts(decom, range(svd.get_n_components()), text)
    ##plt.show()
    print(cos)
    #print(ts1.tfidf_)
    svd = SVD()
    ts.tfidf_ = svd.fit_transform(ts1.tfidf_)
    print(ts._cossimilarity())