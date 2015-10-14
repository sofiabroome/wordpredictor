
"""Abstract Class to represent a ngram model implementation"""


class AbstractNgramModel():

    # We want to train on a corpora
    def train(self, words):
        raise NotImplementedError

    # We want to be able to smooth the model
    def smooth(self, *args):
        raise NotImplementedError

    # We want to be able to predict the next words based on a sentence
    def predict_words(self, sentence, n=10):
        raise NotImplementedError
