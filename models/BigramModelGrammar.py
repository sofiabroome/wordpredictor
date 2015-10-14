from abstract_ngram_model import AbstractNgramModel
import nltk
import random


"""Bigram model implementation"""


class BigramModelGrammar(AbstractNgramModel):

    def train(self, taggedWords):
        self.taggedWords = taggedWords  # each word comes with a tag
        self.wordTagBigrams = list(nltk.bigrams(taggedWords))
        # (a[0][1],b[0][1])  where a,b are different words, 0 is the word and 1 is the tag.

        # self.wordFrequencies = nltk.FreqDist(word for (word, tag) in taggedWords)
        # self.tagFrequencies = nltk.FreqDist(tag for (word, tag) in taggedWords)

        # self.word_probs_bg = nltk.MLEProbDist(self.wordFrequencies)
        # self.tag_probs_bg = nltk.MLEProbDist(self.tagFrequencies)

        self.frequencies = nltk.FreqDist(self.wordTagBigrams)   # frequency of wordtagbigrams
        self.probs_wtbg = nltk.MLEProbDist(self.frequencies)

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
                print matches[j]
                base_query_string = base_query_string + " " + matches[j][0][-1]
            else:
                print "There is no likely ngram with these last words. Try some smoothing methods."
        # print "The result is:\n", base_query_string
        return base_query_string

    def predict_next_word(self, base_query_string):
        matches = []
        query = base_query_string.split()
        for (a, b) in self.wordTagBigrams:
            prob = self.probs_wtbg.prob((a, b))
            if a[0] == unicode(query[-1]):
                if a[1] == 'NOUN':
                    noun_follower = b[1]    # should crudely be a '.','ADP' or 'verb'
                    # see my explore_nltk-notebook for graph
                    if noun_follower == '.':
                        prob += 0.000001
                    if noun_follower == 'ADP':
                        prob += 0.000001
                    if noun_follower == 'VERB':
                        prob += 0.000001  # i know that i'm not normalizing :)
                # print bigram, probs_bg.prob(bigram)
                matches.append(((a, b), prob))
        matches.sort(key=lambda x: x[1])
        return matches
