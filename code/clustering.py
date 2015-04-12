__author__ = 'kolenka'


import utils
from gensim.models import Word2Vec
from sklearn.cluster import KMeans
import numpy as np
import nltk.data

class MySentences(object):
    def __init__(self, dirname):
        self.dirname = dirname

    '''
    This tokenizer divides a text into a list of sentences,
    by using an unsupervised algorithm to build a model for
    abbreviation words, collocations, and words that start
    sentences. It must be trained on a large collection of
    plaintext in the target language before it can be used.

    The NLTK data package includes
    a pre-trained Punkt tokenizer for English.
    '''

    def __iter__(self):
        sent_detector = nltk.data.load('tokenizers/punkt/english.pickle')
        for fname in os.listdir(self.dirname):
            text = open(os.path.join(self.dirname, fname)).read()
            yield sent_detector.tokenize(text.strip())

def read_corpus_word2vec():
    '''
    Read all files in collection and return the word to vector model
    '''
    sent_detector = nltk.data.load('tokenizers/punkt/english.pickle')
    sentences = MySentences('../data_clean/')
    word2vecModel = gensim.models.Word2Vec(sentences)

    return word2vecModel


# Define a function to create bags of centroids and bags of cluster centers

def create_bag_of_centroids( wordlist, word_centroid_map, centroid_word_map ):
    #
    # The number of clusters is equal to the highest cluster index
    # in the word / centroid map
    num_centroids = max( word_centroid_map.values() ) + 1
    #
    # Pre-allocate the bag of centroids vector (for speed)
    bag_of_centroids = np.zeros( num_centroids, dtype="float32" )
    bag_of_centers = []
    #
    # Loop over the words in the review. If the word is in the vocabulary,
    # find which cluster it belongs to, and increment that cluster count
    # by one
    for word in wordlist:
        if word in word_centroid_map:
            index = word_centroid_map[word]
            bag_of_centroids[index] += 1
            bag_of_centers.append(centroid_word_map[index])
    #
    # Return the "bag of centroids"
    return (bag_of_centroids, bag_of_centers)

def run_clustering(tags):
    # vector of tags acheaved from other models
    # tags = ['muslim','holy']

    model = read_corpus_word2vec()

    # Set "k" (num_clusters) to be 1/5th of the vocabulary size, or an
    # average of 5 words per cluster
    word_vectors = model.syn0
    num_clusters = 2*tags.shape[0] - 1

    # Initalize a k-means object and use it to extract centroids
    print "Running K means"
    kmeans_clustering = KMeans( n_clusters = num_clusters)
    kmeanFit = kmeans_clustering.fit( word_vectors )
    idx = kmeanFit.labels_
    centers = kmeanFit.cluster_centers_

    # Create a Word / Index dictionary, mapping each vocabulary word to
    # a cluster number
    word_centroid_map = dict(zip( model.index2word, idx ))

    clusterDist = kmeans_clustering.transform( word_vectors )
    cluster_tags = []
    for i in range(0,num_clusters - 1):
        cluster_tags.append(model.index2word[np.argmax(clusterDist[:,i])])

    centroid_word_map = dict(zip(centers, cluster_tags ))

    # Retrun relenavt claster tags
    top_clusters = create_bag_of_centroids( tags, word_centroid_map, centroid_word_map ).bag_of_centers

    print "Bag of centroids tags: \n"
    print top_clusters

    return top_clusters

#    if __name__ == '__main__':




