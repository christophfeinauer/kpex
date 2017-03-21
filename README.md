kpex: Keyphrase extraction in Python (based on a corpus)
=============================================================================

Overview
--------

This simple package does a term-frequency inverse-document-frequency analysis ([tf-idf](https://en.wikipedia.org/wiki/Tf%E2%80%93idf)])  of a text based on a corpus of texts. This means it extracts possible keyphrases from text and corpus and ranks them, using a score that increases with keyphrase frequency in the text and decreases with keyphrase frequency in the corpus. This is basically the simplest unsupervised approach beyond using raw frequencies for ranking. More sophisticated approaches like [TextRank](https://github.com/davidadamojr/TextRank) or supervised approaches using [training sets](https://github.com/snkim/AutomaticKeyphraseExtraction) will perform significatly better.

Dependencies & Installation
---------------------------

Needed Python packages: nltk, gensim and scikit-learn

Install with:
```
pip install nltk gensim scikit-learn
```

Install kpex:

```
pip install git+https://github.com/christophfeinauer/kpex
```
