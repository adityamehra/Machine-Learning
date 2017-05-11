from __future__ import division

import gensim
import gensim.models.word2vec as w2v
import logging
import multiprocessing
import nltk
from nltk.tokenize import word_tokenize
import numpy as np
import os
from os import listdir
import pickle
import string
import sklearn.manifold
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import cross_val_score
from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import time
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
model = w2v.Word2Vec.load("../trained-models/model.w2v")
train = pd.read_csv("../data/SMSSpamCollection.csv", sep = "\t", header = None)
train.columns = ["label", "message"]
train['label'] = train.label.map({'ham':0, 'spam':1})

X_train, X_test, y_train, y_test = train_test_split(train['message'],
                                                    train['label'],
                                                    random_state=1)

# Set "k" (num_clusters) to be 1/5th of the vocabulary size, or an
# average of 5 words per cluster
word_vectors = model.wv.syn0
num_clusters = word_vectors.shape[0] // 5

# Initalize a k-means object and use it to extract centroids
kmeans_clustering = KMeans( n_clusters = num_clusters )
idx = kmeans_clustering.fit_predict( word_vectors )

# Create a Word / Index dictionary, mapping each vocabulary word to
# a cluster number
word_centroid_map = dict(zip( model.wv.index2word, idx ))

def create_bag_of_centroids( wordlist, word_centroid_map ):
    # The number of clusters is equal to the highest cluster index
    # in the word / centroid map
    num_centroids = max( word_centroid_map.values() ) + 1
    # Pre-allocate the bag of centroids vector (for speed)
    bag_of_centroids = np.zeros( num_centroids, dtype="float32" )
    # Loop over the words in the review. If the word is in the vocabulary,
    # find which cluster it belongs to, and increment that cluster count
    # by one
    for word in wordlist:
        if word in word_centroid_map:
            index = word_centroid_map[word]
            bag_of_centroids[index] += 1
    # Return the "bag of centroids"
    return bag_of_centroids

print "\n ###### Word2Vec ###### \n "
# Pre-allocate an array for the training set bags of centroids (for speed)
train_centroids = np.zeros( (X_train.size, num_clusters), dtype="float32" )
# Transform the training set reviews into bags of centroids
counter = 0
for message in X_train:
    train_centroids[counter] = create_bag_of_centroids( message, word_centroid_map )
    counter += 1

test_centroids = np.zeros( (X_test.size, num_clusters), dtype="float32" )

counter = 0
for message in X_test:
    test_centroids[counter] = create_bag_of_centroids( message, word_centroid_map )
    counter += 1

# Fitting the forest may take a few minutes
print "\nRandom forest..."
forest = RandomForestClassifier(n_estimators = 100)
forest = forest.fit(train_centroids,y_train)
predictions = forest.predict(test_centroids)
print 'Accuracy score: ', accuracy_score(y_test, predictions)
print 'Precision score: ', precision_score(y_test, predictions)
print 'Recall score: ', recall_score(y_test, predictions)
print 'F1 score: ', f1_score(y_test, predictions)

print "\nMultinomial Naive Bayes..."
mnb = MultinomialNB()
mnb = mnb.fit(train_centroids,y_train)
predictions = mnb.predict(test_centroids)
print 'Accuracy score: ', accuracy_score(y_test, predictions)
print 'Precision score: ', precision_score(y_test, predictions)
print 'Recall score: ', recall_score(y_test, predictions)
print 'F1 score: ', f1_score(y_test, predictions)

print "\n ###### Bag of words ###### \n "

vectorizer = CountVectorizer()

train_data_features = vectorizer.fit_transform(X_train)
train_data_features = train_data_features.toarray()

test_data_features = vectorizer.transform(X_test)
test_data_features = test_data_features.toarray()

print "\n Random forest...\n "
forest = RandomForestClassifier(n_estimators = 100)
forest = forest.fit(train_data_features,y_train)
predictions = forest.predict(test_data_features)
print 'Accuracy score: ', accuracy_score(y_test, predictions)
print 'Precision score: ', precision_score(y_test, predictions)
print 'Recall score: ', recall_score(y_test, predictions)
print 'F1 score: ', f1_score(y_test, predictions)

print "\n Multinomial Naive Bayes...\n "
mnb = MultinomialNB()
mnb = mnb.fit(train_data_features,y_train)
predictions = mnb.predict(test_data_features)
print 'Accuracy score: ', accuracy_score(y_test, predictions)
print 'Precision score: ', precision_score(y_test, predictions)
print 'Recall score: ', recall_score(y_test, predictions)
print 'F1 score: ', f1_score(y_test, predictions)
