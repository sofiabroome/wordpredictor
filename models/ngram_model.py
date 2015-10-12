from abstract_ngram_model import AbstractNgramModel
import nltk


"""General Ngram model implementation"""


class NgramModel(AbstractNgramModel):

    def __init__(self, n=4):
        self.n = n

    def train(self, words):
        self.words = words
        self.frequencies = nltk.FreqDist(nltk.ngrams(words, self.n))
        self.probs_bg = nltk.MLEProbDist(self.frequencies)

    def predict_next_word(self, base_query_string):
        matches = []
        query = base_query_string.split()
        for ngram in nltk.ngrams(self.words, self.n):
            if self.equals(ngram, query):
                # print ngram, probs_bg.prob(ngram)
                matches.append((ngram, self.probs_bg.prob(ngram)))
        matches.sort(key=lambda x: x[1])
        return matches

    def equals(self, ngram, query):
        for i in range(self.n - 1):
            if ngram[1 + i] != unicode(query[- self.n + 1 + i]):  # i - self.n + 1 = -1 - (self.n - 1) + (i + 1)
                return False
        return True
