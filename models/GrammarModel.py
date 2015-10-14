from abstract_ngram_model import AbstractNgramModel
import nltk
from bigram_model import BigramModel
from trigram_model import TrigramModel
from ngram_model import NgramModel

"""Grammar Ngram model implementation"""


class GrammarModel(AbstractNgramModel):

    def __init__(self, n=4):
        self.n = n
    
    
    # Should receive words with tags
    def train(self, words):
        tags = []
        for i in range(len(words)):
            tags.append(words[i][1])
        self.ngrams = list(nltk.ngrams(tags, self.n))
        self.frequencies = nltk.FreqDist(self.ngrams)
        self.probs_ng = nltk.MLEProbDist(self.frequencies)
        print self.probs_ng

    def smooth(self, smoothing, *args):
        print self.frequencies
        self.probs_ng = getattr(nltk, smoothing)(*args)
        print self.probs_ng
        
    def predict_next_tag(self, base_query_string):
        # The equivalent of predict_next_word for the other models
        matches = []
        query = base_query_string.split()
        for ngram in self.ngrams:
            if self.at_the_end(ngram, query):
                # print ngram, probs_bg.prob(ngram)
                matches.append((ngram, self.probs_ng.prob(ngram)))
        matches.sort(key=lambda x: x[1])
        return matches

    def predict_next_word(self, model, base_query_string):
        # Returns the same matches than the argument model whith a probability which
        # takes into account the grammar tag of the words of the base_query_string
        
        # First, get the tag sequence associated to the word sequence
        text = nltk.word_tokenize(base_query_string)
        tagged = nltk.pos_tag(text)
        
        # Transform it according to the universal tagset
        simplified = [nltk.map_tag('en-ptb', 'universal', tag) for word, tag in tagged]
        blank = " "
        simple = blank.join(simplified)
        
        # Predict the possible tags and their probabilities after this tag sequence 
        # according to our Grammar Ngrams model
        tag_matches = self.predict_next_tag(simple)
        
        # Keep only the predicted tag (the last one) of each match
        tags = [(tuple[-1],prob) for tuple, prob in list(set(tag_matches))]
        
        # Predict the possible tags and their probabilities after the word sequence 
        # according the Word Ngrams model argument of this method
        word_matches = model.predict_next_word(base_query_string)
        
        # Keep only the predicted word (the last one) of each match
        words = [(tuple[-1],prob) for tuple, prob in list(set(word_matches))]
        
        # Fetch the probability of the tag assoiated to each predicted world
        prior = []
        for i in range(len(words)):
            word, prob = words[i]
            
            # Tag the predicted word
            tag = nltk.pos_tag(nltk.word_tokenize(word))
            universal_tag = nltk.map_tag('brown', 'universal', tag[0][1])
            
            # Take the probabilities of the equal tag in our GrammarModel predictions or 0
            # if this tag is not possible after our tag sequence
            p = 0
            for j in range(len(tags)):
                if universal_tag == tags[j][0]:
                    p = tags[j][1]
            prior.append(p)
        
        # Return the same matches than our Word NgramModel before, with a new probability
        better_matches = []
        for i in range(len(prior)):
            tuple, prob = list(set(word_matches))[i]
            better_matches.append((tuple, prob*prior[i]))
        return better_matches

    def at_the_end(self, ngram, query):
        for i in range(self.n - 1):
            if ngram[i] != unicode(query[- self.n + 1 + i]):
                return False
        return True
