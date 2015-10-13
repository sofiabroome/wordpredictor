#!/usr/bin/env python
import nltk
words = nltk.corpus.gutenberg.words('melville-moby_dick.txt')
for w in words:
    print w
