
"""Abstract Class to represent a ngram model implementation"""


class AbstractNgramModel():

    def train(self, words):
        raise NotImplementedError

    def predict_next_word(self, base_query_string):
        raise NotImplementedError

    def predict_words(self, base_query_string, n=100):
        raise NotImplementedError
