from math import*
from decimal import Decimal

class Similarity(object):
    @staticmethod
    def euclidean_distance(x,y):
        return sqrt(sum(pow(a-b,2) for a, b in zip(x, y)))

    @staticmethod
    def manhattan_distance(x,y):
        return sum(abs(a-b) for a,b in zip(x,y))

    @staticmethod
    def minkowski_distance(x,y,p_value):
        return Similarity.nth_root(sum(pow(abs(a-b),p_value) for a,b in zip(x, y)),
           p_value)

    @staticmethod
    def nth_root(value, n_root):
        """ returns the n_root of an value """
        root_value = 1/float(n_root)
        return round (Decimal(value) ** Decimal(root_value),3)

    @staticmethod
    def square_rooted(x):
        """ return 3 rounded square rooted value """
        return round(sqrt(sum([a*a for a in x])),3)

    @staticmethod
    def cosine_similarity(x,y):
        numerator = sum(a*b for a,b in zip(x,y))
        denominator = Similarity.square_rooted(x)*Similarity.square_rooted(y)
        return round(numerator/float(denominator),3)

    @staticmethod
    def jaccard_similarity(x,y):
        intersection_cardinality = len(set.intersection(*[set(x), set(y)]))
        union_cardinality = len(set.union(*[set(x), set(y)]))
        return intersection_cardinality/float(union_cardinality)
