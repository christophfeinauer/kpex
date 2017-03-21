import nltk
import string
import itertools
import gensim
import math
import errno
import os

def read_txt(file):
    with open(file) as fid:
        doc = fid.read();
    return doc

def extract_chunks(text_string,max_words=3,lemmatize=False):

    # Any number of adjectives followed by any number of nouns and (optionally) again
    # any number of adjectives folowerd by any number of nouns
    grammar=r'KT: {(<JJ>* <NN.*>+ <IN>)? <JJ>* <NN.*>+}'

    # Makes chunks using grammar regex
    chunker = nltk.RegexpParser(grammar)
    
    # Get grammatical functions of words
    # What this is doing: tag(sentence -> words)
    tagged_sents = nltk.pos_tag_sents(nltk.word_tokenize(sent) for sent in nltk.sent_tokenize(text_string))
    
    # Make chunks from the sentences, using grammar. Output in IOB.
    all_chunks = list(itertools.chain.from_iterable(nltk.chunk.tree2conlltags(chunker.parse(tagged_sent))
                                                        for tagged_sent in tagged_sents))
    # Join phrases based on IOB syntax.
    candidates = [' '.join(w[0] for w in group).lower() for key, group in itertools.groupby(all_chunks, lambda l: l[2] != 'O') if key]
    
    # Filter by maximum keyphrase length
    candidates = list(filter(lambda l: len(l.split()) <= 3, candidates))
    
    # Filter phrases consisting of punctuation or stopwords
    punct = set(string.punctuation)
    stop_words = set(nltk.corpus.stopwords.words('english'))
    candidates = list(filter(lambda l: l not in stop_words and not all(c in punct for c in l),candidates))
    
    # lemmatize
    if lemmatize:
        lemmatizer = nltk.stem.WordNetLemmatizer().lemmatize
        candidates =  [lemmatizer(x) for x in candidates]

    return candidates

def extract_terms_with_corpus_sklearn(text_files, number_of_terms=10, max_features=20, max_words=3, lemmatize=True, train_on_script = True):
    
    # check input
    for file in text_files:
        if not os.path.isfile(file):
            raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), file)
    if number_of_terms>max_features:
        raise Exception("number_of_terms has to be smaller than max_features")

    # tokenizer
    analyzer = lambda s: extract_chunks(read_txt(s),lemmatize=lemmatize,max_words=max_words)

    # All-in-one object for tfidf calculation
    tfidf_vectorizer = TfidfVectorizer(input='filename', analyzer = analyzer, max_features=max_features)
    
    # fit training data & get tfidf matrix
    if train_on_script:
        tfidf_mat = tfidf_vectorizer.fit(text_files[0:])
    else: 
        tfidf_mat = tfidf_vectorizer.fit(text_files[1:])
    
    # transform first file
    tfidf_script = tfidf_vectorizer.transform([text_files[0]])
    
    # get map between id and term
    id2term = tfidf_vectorizer.get_feature_names()

    return [(id2term[i],tfidf_script[0,i]) for i in tfidf_script.toarray()[0,:].argsort()[::-1][0:number_of_terms]]


def extract_terms_with_corpus_gensim(text_files,number_of_terms=5,max_words=3, lemmatize=True, filter_no_below=0, filter_no_above=1.0):
    
    chunked_texts = [extract_chunks(read_txt(text),max_words=max_words,lemmatize=lemmatize) for text in text_files]
    
    dictionary = gensim.corpora.Dictionary(chunked_texts)
    #dictionary.filter_extremes(no_below=filter_no_below, no_above=filter_no_above)
    
    # Bag of words representation of the text
    cp = [dictionary.doc2bow(boc_text) for boc_text in chunked_texts]
    
    # tf/idf frequency model
    tf = gensim.models.TfidfModel(cp[0:],normalize=False,wglobal = lambda df,D: math.log((1+D)/(1+df))+1)
    
    # transform script
    ts = tf[cp][0]
    
    # sort by score
    s = sorted(ts,key=lambda ts: ts[1], reverse=True)
    
    # retranslate in terms
    terms = [(dictionary[s[0]],s[1]) for s in s]
    return terms[0:number_of_terms]

def test():
    text_files = ['../dat/script.txt','../dat/transcript_1.txt','../dat/transcript_2.txt','../dat/transcript_3.txt']
    terms = extract_terms_with_corpus(text_files)
    return terms
