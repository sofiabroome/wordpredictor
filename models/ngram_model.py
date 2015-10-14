from abstract_ngram_model import AbstractNgramModel
import nltk


"""General Ngram model implementation"""


class NgramModel(AbstractNgramModel):

    def __init__(self, n=4):
        self.n = n

    def train(self, words):
        self.words = words
        self.ngrams = list(nltk.ngrams(words, self.n))
        self.frequencies = nltk.FreqDist(self.ngrams)
        self.probs_ng = nltk.MLEProbDist(self.frequencies)

    def predict_next_word(self, base_query_string):
        matches = []
        query = base_query_string.split()
        for ngram in self.ngrams:
            if self.at_the_end(ngram, query):
                # print ngram, probs_bg.prob(ngram)
                matches.append((ngram, self.probs_ng.prob(ngram)))
        matches.sort(key=lambda x: x[1])
        return matches

    def at_the_end(self, ngram, query):
        for i in range(self.n - 1):
            if ngram[i] != unicode(query[- self.n + 1 + i]):
                return False
        return True
