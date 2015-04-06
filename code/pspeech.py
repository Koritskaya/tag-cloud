import nltk
import numpy as np
import sys

def read_info(infile="../data/info.tsv"):
    '''Read in info.tsv and output a list of lists of the form:
    [[doc_path, president, title, data],...]'''
    result = []
    with open(infile, "r") as content:
        for line in content:
            line = line.strip("\n")
            result.append(line.split("\t"))
    return result
    
def read_corpus():
    '''
    Read all files in collection and return the tf count. The output array
    has a shape: nr_of_docs x vocabulary_size. This function also returns
    the vocalubary which is a list of words.
    '''
    pattern = r'''(?x)               # set flag to allow verbose regexps
              ([A-Z]\.)+         # abbreviations, e.g. U.S.A.
              | \$?\d+(\.\d+)?%? # numbers, incl. currency and percentages
              | \w+([-']\w+)*    # words w/ optional internal hyphens/apostrophe
              | [+/\-@&*]        # special characters with meanings
            '''
    
    tokenizer = nltk.tokenize.RegexpTokenizer(pattern) # this will only keep words
    stemmer = nltk.stem.porter.PorterStemmer();
    info = read_info() # [[doc_path, president, title, data],...]
    VOCAB_SIZE = int(1e5) # trim this later but ideally we should grow this
    NUM_DOCS = len(info) # get number of docs
    vocab = list()
    tf = np.zeros((NUM_DOCS, VOCAB_SIZE))
    for indDoc in range(0, NUM_DOCS):
        # read file
        infile = open(info[indDoc][0], 'r').read()
        infile = infile.replace('&#39;', "'")
        infile =  infile.replace('&mdash;', "-")
        
        tokens = tokenizer.tokenize(infile)
        for token in tokens:
            try:
                new_token = token.lower() #stemmer.stem(token.lower())
                try:
                    ind = vocab.index(new_token) # get the index in the vocabulary
                except ValueError:
                    vocab.append(new_token) # add to the vocabulary
                    ind = len(vocab) 
                tf[indDoc, ind]+=1 # increment the count for this doc
            except UnicodeDecodeError:
                continue
        # progress
        sys.stdout.write("\rReading collection: %d%%" %(indDoc*100/NUM_DOCS))
        sys.stdout.flush()     
           
    tf = tf[:,0:len(vocab)] # trim the sparse matrix with counts
    return (vocab, tf)

def tfidf(tf):
    '''
    Implement classical TF*IDF.
    Inputs
    tf -- array with tf counts, rows are documents, columns are vocabulary
    
    Outputs
    weights -- td*idf weights
    '''
    N = tf.shape[0] # number of docs in the corpus
    appCorpus = np.sum(tf>0, 0) # get the nr of documents where the word appears
    idf = np.log(float(N)/appCorpus)
    weights = np.zeros(tf.shape) # init array for weights
    for i in range(0, tf.shape[0]):
        weights[i,:] = tf[i,:]*idf
    return weights

def filter_speech(filt, filt_type='president'):
    info = read_info()
    result = []
    for indDoc in range(0, len(info)):
        if filt_type=='president' and info[indDoc][1].lower().count(filt.lower()):
            result.append(indDoc)
        if filt_type=='year' and info[indDoc][3].lower().count(filt.lower()):
            result.append(indDoc)
    return result
            
def get_tags(vocab, weights, indDoc, topN=20):
    '''
    Given a vocabulary and weights return top N words for the document 
    with indec indDoc
    '''
    sortW = weights[indDoc,:].argsort()
    result = [vocab[i] for i in reversed(sortW[-topN:])]
    return result
    
    
if __name__ == '__main__':
    (vocab,tf) = read_corpus()
    weights = tfidf(tf)
    topWords = get_tags(vocab, weights, 0) # get top 20 words for Obama's Acceptance Speech
    print topWords

