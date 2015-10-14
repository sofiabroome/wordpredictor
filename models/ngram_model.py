from abstract_ngram_model import AbstractNgramModel
import nltk
import random


"""General Ngram model implementation"""


class NgramModel(AbstractNgramModel):

    def __init__(self, n=4):
        assert n != 1
        self.n = n

    def train(self, words):
        self.words = words
        self.unique_words = [w[0] for w in set(nltk.ngrams(words, 1))]
        self.ngrams = list(nltk.ngrams(words, self.n))
        self.frequencies = nltk.FreqDist(self.ngrams)
        self.probs_ng = nltk.MLEProbDist(self.frequencies)

    def smooth(self, smoothing, *args):
        self.probs_ng = getattr(nltk, smoothing)(*args)

    def predict_words(self, base_query_string, n=10):
        for i in range(n):
            matches = self.predict_next_word(base_query_string)
            prob_sum = sum(m[-1] for m in matches)
            matches = [[m[0], m[-1] / prob_sum] for m in matches]
            rand = random.random()
            j = 0
            if len(matches) != 0:
                while (rand > matches[j][-1]):
                    rand -= matches[j][-1]
                    j += 1
                # print "Winning match is: ", matches[j]
                base_query_string = base_query_string + " " + matches[j][0][-1]
            else:
                print "There is no likely ngram with these last words. Try some smoothing methods."
                break
        return base_query_string

    def predict_next_word(self, base_query_string):
        matches = []
        query = base_query_string.split()[-(self.n - 1):]
        query.append("")
        for w in self.unique_words:
            query[-1] = w
            ngram = tuple(query)
            prob = self.probs_ng.prob(ngram)
            if prob > 0.0:
                # print ngram, probs_bg.prob(ngram)
                matches.append((ngram, prob))
        matches.sort(key=lambda x: x[1])
        return matches
