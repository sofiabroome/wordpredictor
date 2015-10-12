from abstract_ngram_model import AbstractNgramModel
import nltk


"""Trigram model implementation"""


class TrigramModel(AbstractNgramModel):

    def train(self, words):
        self.words = words
        self.trigrams = nltk.trigrams(words)
        self.frequencies = nltk.FreqDist(self.trigrams)
        self.trigrams = nltk.trigrams(words)
        self.probs_bg = nltk.MLEProbDist(self.frequencies)

    def predict_next_word(self, query_string):
        matches = []
        query = query_string.split()
        for trigram in self.trigrams:
            if trigram[0] == unicode(query[-2])\
               and trigram[1] == unicode(query[-1]):
                # print trigram, probs_bg.prob(trigram)
                matches.append((trigram, self.probs_bg.prob(trigram)))
        matches.sort(key=lambda x: x[1])
        return matches
