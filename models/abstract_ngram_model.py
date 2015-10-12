import random

"""Abstract Class to represent a ngram model implementation"""


class AbstractNgramModel():

    def train(self, words):
        raise NotImplementedError

    def predict_next_word(self, base_query_string):
        raise NotImplementedError

    def predict_words(self, base_query_string, n=100):
        matches = []
        for i in range(n):
            matches = self.predict_next_word(base_query_string)
            weight_sum = sum(m[-1] for m in matches)
            norm_matches = [(m[0], m[-1] / weight_sum) for m in matches]
            rand = random.random()
            i = 0
            if len(norm_matches) != 0:
                while (rand > norm_matches[i][-1]):
                    rand -= norm_matches[i][-1]
                    i += 1
                base_query_string = base_query_string + " " + norm_matches[i][0][-1]
            else:
                print "NO_MORE_OPTION"
        print "The result is:\n", base_query_string
        return base_query_string
