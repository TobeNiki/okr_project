import MeCab
from TextSimlarity import TextSimilarity

class Morpheme:
    def __init__(self):
        self.tagger = MeCab.Tagger('-Ochasen')
        
    def fit_transform(self, corpus):
        result = [] 
        for sentence in corpus:
            text = ""
            node = self.tagger.parseToNode(sentence)
            while node:
                text = text + node.surface + " "
                """
                if node.feature.split(",")[0] == "名詞":
                    meishi_count = meishi_count + 1
                    meishi_list.append(node.surface)
                elif node.feature.split(",")[0] == "動詞":
                    doushi_count = doushi_count + 1
                    doushi_list.append(node.surface)
                else:pass
                """
                node = node.next
            result.append(text)
        #return [self.tagger.parse(sentence).strip() for sentence in corpus]
        return result
    
tagger = MeCab.Tagger('-Ochasen')

def tokenize( text:str)->list:
    tokens = []
    node = tagger.parseToNode(text)
    while node:
        if node.surface != '':
            tokens.append(node.surface)
        node.next
    return tokens

if __name__ == "__main__":
    morpheme = Morpheme()
    text = [
        'このコードではMeCab.Taggerクラスのインスタンスを生成。',
        'そのparseメソッドに「くるまでまつ」という文字列を渡すことで解析を実行しています。',
        'その実行結果は次のようになりました。',
        'MeCab.Taggerクラスのインスタンス生成を行っている行にはコメントがあります。',
        '形態素解析実行時に参照する辞書を変更するといったことが可能です。'
    ]
    corpus = morpheme.fit_transform(text)
    print(corpus)
    ts = TextSimilarity()
    cos = ts.fit_transform(corpus)
    print(type(cos.tolist()),type(ts.get_feature_name()),type(ts.get_tfidf().tolist()))
    print(ts.get_tfidf().tolist(),cos.tolist())