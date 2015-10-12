from abstract_ngram_model import AbstractNgramModel
import nltk


"""Bigram model implementation"""


class BigramModel(AbstractNgramModel):

    def train(self, words):
        self.words = words
        self.bigrams = nltk.bigrams(words)
        self.frequencies = nltk.FreqDist(self.bigrams)
        self.bigrams = nltk.bigrams(words)
        self.probs_bg = nltk.MLEProbDist(self.frequencies)

    def predict_next_word(self, query_string):
        matches = []
        query = query_string.split()
        for bigram in self.bigrams:
            if bigram[0] == unicode(query[-1]):
                # print bigram, probs_bg.prob(bigram)
                matches.append((bigram, self.probs_bg.prob(bigram)))
        matches.sort(key=lambda x: x[1])
        return matches
