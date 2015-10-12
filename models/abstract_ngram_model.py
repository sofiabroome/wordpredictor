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
            prob_sum = sum(m[-1] for m in matches)
            rand = random.random() * prob_sum
            j = 0
            if len(matches) != 0:
                while (rand > matches[j][-1]):
                    rand -= matches[j][-1]
                    j += 1
                # print "Winning match is: ", norm_matches[j]
                base_query_string = base_query_string + " " + matches[j][0][-1]
            else:
                print "There is no likely ngram with these last words. Try some smoothing methods."
        # print "The result is:\n", base_query_string
        return base_query_string
