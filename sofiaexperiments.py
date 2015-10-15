from models.bigram_model import BigramModel
from models.trigram_model import TrigramModel
from models.ngram_model import NgramModel

from nltk.corpus import gutenberg

gutenberg.fileids()

def test_corpora(corpora, query, n):
    for k in range(2, 5, 1):
        fm = NgramModel(k)
        fm.train(corpora)
        print "Result for a",(str(k)+"-gram: "),fm.predict_words(query, n)

t1 = gutenberg.words('bible-kjv.txt');
t2 = gutenberg.words('whitman-leaves.txt');
t3 = gutenberg.words('blake-poems.txt');
print len(t1), len(t2), len(t3)


# for k in range(2, 5, 1):
#     fm1 = NgramModel(k)
#     fm1.train(t1)
#     print "Testing with the bible (very large corpus)"
#     print "Results for a",(str(k)+"-gram: ")
#     for k in range(5):
#         print fm1.predict_words("I am afraid they", 10)
#     print "\n\n"


for k in range(2, 5, 1):
    fm2 = NgramModel(k)
    fm2.train(t2)
    print "Testing with the poetry collection Leaves of Grass, medium large (154883 words)"
    print "Results for a",(str(k)+"-gram: ")
    for k in range(5):
        print fm2.predict_words("I am afraid they", 10)
    print "\n\n"

# for k in range(2, 5, 1):
#     fm3 = NgramModel(k)
#     fm3.train(t3)
#     print "Testing with Blake poems, small corpus (8354 words)"
#     print "Results for a",(str(k)+"-gram: ")
#     for k in range(5):
#         print fm3.predict_words("I am afraid they", 10)
#     print "\n\n"