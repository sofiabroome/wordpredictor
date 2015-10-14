#!/usr/bin/env python
import nltk
sents = nltk.corpus.gutenberg.sents(fileids='melville-moby_dick.txt')
for sent in sents:
    print ' '.join(sent)

