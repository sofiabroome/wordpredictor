
"""Abstract Class to represent a ngram model implementation"""


class AbstractNgramModel():

    def train(self, words):
        raise NotImplementedError

    def predict_words(self, n=100):
        raise NotImplementedError
