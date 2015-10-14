from models.bigram_model import BigramModel
from models.trigram_model import TrigramModel
from models.ngram_model import NgramModel
from models.BigramModelGrammar import BigramModelGrammar
import nltk

from nltk.corpus import brown

brown_fiction_tagged = brown.tagged_words(categories='fiction', tagset='universal')
tag_fd = nltk.FreqDist(tag for (word, tag) in brown_fiction_tagged)

bmg = BigramModelGrammar()
bmg.train(brown_fiction_tagged)

matches = bmg.predict_next_word("the landlord came into the room")
print matches[-1]