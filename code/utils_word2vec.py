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

def sentence_process(sentence):

    '''
    common100 = {1: 'the', 2: 'of', 3: 'and', 4: 'a', 5: 'to', 6: 'in', 7: 'is', 8: 'you', 9: 'that', 10: 'it',
    11: 'he', 12: 'was', 13: 'for', 14: 'on', 15: 'are', 16: 'as', 17: 'with', 18: 'his', 19: 'they',
    20: 'I', 21: 'at', 22: 'be', 23: 'this', 24: 'have', 25: 'from', 26: 'or', 27: 'one', 28: 'had', 29: 'by',
    30: 'word', 31: 'but', 32: 'not', 33: 'what', 34: 'all', 35: 'were', 36: 'we', 37: 'when', 38: 'your', 39: 'can',
    40: 'said', 41: 'there', 42: 'use', 43: 'an', 44: 'each', 45: 'which', 46: 'she', 47: 'do', 48: 'how', 49: 'their',
    50: 'if', 51: 'will', 52: 'up', 53: 'other', 54: 'about', 55: 'out', 56: 'many', 57: 'then', 58: 'them', 59: 'these',
    60: 'so', 61: 'some', 62: 'her', 63: 'would', 64: 'make', 65: 'like', 66: 'him', 67: 'into', 68: 'time', 69: 'has',
    70: 'look', 71: 'two', 72: 'more', 73: 'write', 74: 'go', 75: 'see', 76: 'number', 78: 'way', 79: 'could',
    80: 'people', 81: 'my', 82: 'than', 83: 'first', 84: 'water', 85: 'been', 86: 'call', 87: 'who', 88: 'oil', 89: 'its',
    90: 'now', 91: 'find', 92: 'long', 93: 'down', 94: 'day', 95: 'did', 96: 'get', 97: 'come', 98: 'made', 99: 'may', 100: 'part'}
    '''
    common100 = {1: 'the', 2: 'of', 3: 'and', 4: 'a', 5: 'to', 6: 'in', 7: 'is', 8: 'you', 9: 'that', 10: 'it',
    11: 'he', 13: 'for', 14: 'on', 15: 'are', 16: 'as', 17: 'with', 18: 'his', 19: 'they',
    20: 'I', 21: 'at', 22: 'be', 23: 'this', 25: 'from', 26: 'or', 27: 'one', 29: 'by',
    30: 'word', 31: 'but', 32: 'not', 33: 'what', 34: 'all', 35: 'were', 36: 'we', 37: 'when', 38: 'your', 39: 'can',
     41: 'there', 43: 'an', 45: 'which', 46: 'she', 47: 'do', 48: 'how', 49: 'their',
    50: 'if', 51: 'will', 52: 'up', 53: 'other', 54: 'about', 55: 'out', 56: 'many', 57: 'then', 58: 'them', 59: 'these',
    60: 'so', 61: 'some', 62: 'her', 63: 'would', 66: 'him', 67: 'into', 69: 'has',
    82: 'than', 85: 'been', 87: 'who', 89: 'its', 90: 'Bless'}

    common100 = common100.values()

    pattern = r'''(?x)               # set flag to allow verbose regexps
              ([A-Z]\.)+         # abbreviations, e.g. U.S.A.
              | \$?\d+(\.\d+)?%? # numbers, incl. currency and percentages
              | \w+([-']\w+)*    # words w/ optional internal hyphens/apostrophe
              | [+/\-@&*]        # special characters with meanings
            '''
    tokenizer = nltk.tokenize.RegexpTokenizer(pattern) # this will only keep words

    tokens = []
    this = tokenizer.tokenize(sentence)
    if len(this)>0:
        this[0] = this[0].lower()
    tokens = tokens + this

    # Disregard the most commom words in English
    res = []
    for token in tokens:
        if not(common100.count(token)):
            res.append(token)
    tokens = res
    return tokens

def read_corpus_word2vec():
    info = utils.read_info("../info.tsv")
    '''
    This tokenizer divides a text into a list of sentences,
    by using an unsupervised algorithm to build a model for
    abbreviation words, collocations, and words that start
    sentences. It must be trained on a large collection of
    plaintext in the target language before it can be used.

    The NLTK data package includes
    a pre-trained Punkt tokenizer for English.
    '''
    sent_detector = nltk.data.load('tokenizers/punkt/english.pickle')
    sentences = []
    NUM_DOCS = len(info) # get number of docs
    for indDoc in range(0, NUM_DOCS):
        text = open(info[indDoc][0]).read()
        raw_sentences = sent_detector.tokenize(text.strip())
        for raw_sentence in raw_sentences:
            if len(raw_sentence) > 0:
                sentences.append(sentence_process(raw_sentence))

        sys.stdout.write("\rReading collection: %d%%" %(indDoc*100/NUM_DOCS))
        sys.stdout.flush()
    return sentences


def write_model_word2vec():
    '''
    Read all files in collection and return the word to vector model
    '''

    sentences = read_corpus_word2vec()# MySentences('../data_clean/')
    word2vecModel = gensim.models.Word2Vec(sentences)
    word2vecModel.save('../word2vec/mymodel')
    print word2vecModel.syn0.shape


def read_word2vec():
    '''
    Read all files in collection and return the word to vector model
    '''

    word2vecModel = gensim.models.Word2Vec.load('../word2vec/mymodel')

    return word2vecModel

if __name__ == '__main__':
    print "Reading corpus and building model"
    write_model_word2vec()
