import utils
import numpy as np

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
    weights = np.zeros(tf.shape, dtype='float') # init array for weights
    for i in range(0, tf.shape[0]):
        weights[i,:] = tf[i,:]*idf
    return weights
    
def bm25(tf):
    k1 = 1.6
    b = 0.75
    N = tf.shape[0] # number of docs in the corpus
    avgD = np.sum(tf)/float(N)
    appCorpus = np.sum(tf>0, 0) # get the nr of documents where the word appears
    idf = np.log((float(N) - appCorpus + 0.5)/(appCorpus + 0.5))
    #idf = np.log(float(N)/appCorpus)
    weights = np.zeros(tf.shape, dtype='float') # init array for weights
    for i in range(0, tf.shape[0]):
        lengthD = np.sum(tf[i,:])
        weights[i,:] = idf * tf[i,:] * (k1+1) / (tf[i,:] + k1 * (1 - b + b * lengthD/avgD))
    return weights

def run_model(model_name, vocab, tf):
    #(vocab,tf) = utils.read_corpus()
    N = len(tf) # number of docs in corpus
    weights = []
    if model_name == "bm25":
        weights = bm25(tf)
    elif model_name == "tfidf":
        weights = tfidf(tf)

    top_words = []
    for i in range(N):
        gen_tags = utils.get_tags(vocab, weights, i)
        top_words += gen_tags
    return top_words

#if __name__ == '__main__':
    #run_model(model_name, params)
