import MeCab
import sys
"""
for path in sys.path:
    print(path)
print("==========")
"""
import os

path = os.getcwd()

print(path)
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
    