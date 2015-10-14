from models.bigram_model import BigramModel
from models.trigram_model import TrigramModel
from models.ngram_model import NgramModel
from models.BigramModelGrammar import BigramModelGrammar
from models.GrammarModel import GrammarModel
import nltk
from nltk.corpus import brown

g = GrammarModel()
g.train(nltk.corpus.brown.tagged_words(tagset='universal'))
t = TrigramModel()
t.train(nltk.corpus.brown.words())
query_string = "And now for something"
better = g.predict_next_word(t,query_string)
print better