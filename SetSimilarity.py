from typing import List
class SetSimilarity:
    @staticmethod
    def jaccard_similarity_coefficient(list_a:list, list_b:list)->float:
        set_intersection = set.intersection(set(list_a), set(list_b))
        num_intersection = len(set_intersection)
        set_union = set.union(set(list_a), set(list_b))
        num_union = len(set_union)
        try:
            return float(num_intersection)/num_union
        except ZeroDivisionError:
            return 1.0
    @staticmethod
    def dice_similarity_coefficient(list_a:list, list_b:list)->float:
        set_intersection = set.intersection(set(list_a), set(list_b))
        num_intersection = len(set_intersection)
        num_listA, num_listB = len(list_a), len(list_b)
        try:
            return float(2.0 * num_intersection) / (num_listA + num_listB)
        except ZeroDivisionError:
            return 1.0
    @staticmethod
    def overlap_coefficient(list_a:list, list_b:list)->float:#集合の要素数に条件
        set_intersection = set.intersection(set(list_a), set(list_b))
        num_intersection = len(set_intersection)
        num_listA, num_listB = len(list_a), len(list_b)
        try:
            return float(num_intersection) / min([num_listA, num_listB])
        except ZeroDivisionError:
            return 1.0
    @staticmethod
    def similarity_to_cost(similarity:float)->float:
        return 1.0 - similarity


if __name__ == '__main__':
    list_a = ["リンゴ","ブドウ","イチゴ","パイン","キウイ","メロン"] #集合A
    list_b = ["メロン","イチゴ","リンゴ","パインアップル"] #集合B
 
    overlap = SetSimilarity.overlap_coefficient(list_a,list_b) #Simpson係数を計算
    print(SetSimilarity.similarity_to_cost(overlap)) #計算結果を出力　⇒　0.75
