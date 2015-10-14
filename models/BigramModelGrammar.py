from abstract_ngram_model import AbstractNgramModel
import nltk


"""Bigram model implementation"""


class BigramModelGrammar(AbstractNgramModel):

    def train(self, taggedWords):
        self.taggedWords = taggedWords #each word comes with a tag
        self.wordTagBigrams = list(nltk.bigrams(taggedWords)) # (a[0][1],b[0][1])  where a,b are
                                                              # different words, 0 is the
                                                              # word and 1 is the tag.

        #self.wordFrequencies = nltk.FreqDist(word for (word, tag) in taggedWords)
        #self.tagFrequencies = nltk.FreqDist(tag for (word, tag) in taggedWords)

        #self.word_probs_bg = nltk.MLEProbDist(self.wordFrequencies)
        #self.tag_probs_bg = nltk.MLEProbDist(self.tagFrequencies)

        self.frequencies = nltk.FreqDist(self.wordTagBigrams) #frequency of wordtagbigrams
        self.probs_wtbg = nltk.MLEProbDist(self.frequencies)

    def predict_next_word(self, base_query_string):
        matches = []
        query = base_query_string.split()
        for (a,b) in self.wordTagBigrams:
            prob = self.probs_wtbg.prob((a,b))
            if a[0] == unicode(query[-1]):
                if a[1] == 'NOUN':
                    noun_follower = b[1] #should crudely be a '.','ADP' or 'verb'
                                         # see my explore_nltk-notebook for graph
                    if noun_follower == '.':
                        prob += 0.000001
                    if noun_follower == 'ADP':
                        prob += 0.000001
                    if noun_follower == 'VERB':
                        prob += 0.000001  # i know that i'm not normalizing :) 
                # print bigram, probs_bg.prob(bigram)
                matches.append(((a,b), prob))
        matches.sort(key=lambda x: x[1])
        return matches