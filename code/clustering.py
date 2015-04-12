__author__ = 'kolenka'

import utils
import gensim
import sklearn
from gensim.models import Word2Vec
from sklearn.cluster import KMeans
import numpy as np
import os
import nltk.data
import sys

import utils_word2vec

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
    print "Reading word2vec model"
    model = utils_word2vec.read_word2vec()
    word_vectors = model.syn0
    num_clusters = 2*len(tags) - 1

    print  model.most_similar('iraq')

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
    print clusterDist.shape
    cluster_tags = []
    for i in range(0,num_clusters - 1):
        cluster_tags.append(model.index2word[np.argmax(clusterDist[:,i])])
        mymax = np.argmax(clusterDist[:,i])
#        print np.argmax(clusterDist[:,i])
#        print "word" + model.index2word[mymax]

    centroid_word_map = dict(zip(centers, cluster_tags ))

    # Retrun relenavt claster tags
    top_clusters = create_bag_of_centroids( tags, word_centroid_map, centroid_word_map ).bag_of_centers

    print "Bag of centroids tags: \n"
    print top_clusters

    return top_clusters

#    if __name__ == '__main__':




