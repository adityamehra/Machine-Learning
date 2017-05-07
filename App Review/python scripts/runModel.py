from __future__ import division

import gensim
import gensim.models.word2vec as w2v
import logging
import multiprocessing
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from nltk import pos_tag
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer
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
from sklearn.model_selection import cross_val_score
import time
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

# RUNNAME = "facebook"
# RUNNAME = "clashofclans"
RUNNAME = "allReviews"
# RUNNAME = "gmail"
# RUNNAME = "instagram"

model = w2v.Word2Vec.load("../trained-models3/" + RUNNAME + ".w2v")
train = pd.read_csv( "../processed data/" + RUNNAME + ".csv", header=0, delimiter=",", quoting=2 )

print "\nfix"
print model.most_similar("fix")

print "\nproblem"
print model.most_similar("problem")

# print "\nupdate"
# print reviews2vec.most_similar("update")
# print "freeze"
# print reviews2vec.most_similar("freeze")
# print "\niphone"
# print reviews2vec.most_similar("iphone")
# print "\nhate"
# print reviews2vec.most_similar("hate")
# print len(reviews2vec.wv.vocab)
# print reviews2vec.wv.syn0.shape
# print reviews2vec.wv.index2word
# print len(reviews2vec.wv.index2word)
# print len(set(reviews2vec.wv.index2word))

# Set "k" (num_clusters) to be 1/5th of the vocabulary size, or an
# average of 5 words per cluster
word_vectors = model.wv.syn0
num_clusters = word_vectors.shape[0] // 3

# Initalize a k-means object and use it to extract centroids
kmeans_clustering = KMeans( n_clusters = num_clusters )
idx = kmeans_clustering.fit_predict( word_vectors )

# Create a Word / Index dictionary, mapping each vocabulary word to
# a cluster number
word_centroid_map = dict(zip( model.wv.index2word, idx ))

# print word_centroid_map

# For the first 10 clusters
# for cluster in xrange(0,10):
#     #
#     # Print the cluster number
#     print "\nCluster %d" % cluster
#     #
#     # Find all of the words for that cluster number, and print them out
#     words = []
#     for i in xrange(0,len(word_centroid_map.values())):
#         if( word_centroid_map.values()[i] == cluster ):
#             words.append(word_centroid_map.keys()[i])
#     print words

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
    #
    # Return the "bag of centroids"
    return bag_of_centroids

# Pre-allocate an array for the training set bags of centroids (for speed)
train_centroids = np.zeros( (train["review"].size, num_clusters), dtype="float32" )

# Transform the training set reviews into bags of centroids
counter = 0
for review in train["review"]:
    train_centroids[counter] = create_bag_of_centroids( review, word_centroid_map )
    counter += 1

# Fitting the forest may take a few minutes
print "\nFitting a random forest to labeled training data..."
forest = RandomForestClassifier(n_estimators = 100)
forest = forest.fit(train_centroids,train["helpful"])
result = forest.predict(train_centroids)
score = forest.score(train_centroids,train["helpful"])
print "score is ", score

count = 0
for i in range(len(result)):
    if train["helpful"][i] == result[i]:
        count += 1
print count
print "Percent uncorrectly classified ", ( len(result) - count ) * 100 / len(result )
print "Percent correctly classified ", count * 100 / len( result )

forest = RandomForestClassifier(n_estimators = 100)
scores = cross_val_score( forest, train_centroids, train["helpful"], cv=10 )
print "Scores - \n",  scores
print scores.mean()

# Write the test results
# output = pd.DataFrame(data={"sentiment":result, "review":train["review"]})
# output.to_csv( "BagOfCentroids.csv", index=False, quoting=2 )
print "\nFitting a SVM to labeled training data..."
clf = svm.SVC()
clf = clf.fit(train_centroids,train["helpful"])

result = clf.predict(train_centroids)
score = clf.score(train_centroids,train["helpful"])
print "score is ", score

count = 0
for i in range(len(result)):
    if train["helpful"][i] == result[i]:
        count += 1
print count
print "Percent uncorrectly classified ", ( len(result) - count ) * 100 / len(result )
print "Percent correctly classified ", count * 100 / len( result )

clf = svm.SVC()
scores = cross_val_score( clf, train_centroids, train["helpful"], cv=10 )
print "Scores - \n",  scores
print scores.mean()

print "\nFitting a GaussianNB to labeled training data..."
gnb = GaussianNB()
gnb = gnb.fit(train_centroids, train["helpful"])

result = gnb.predict(train_centroids)
score = gnb.score(train_centroids,train["helpful"])
print "score is ", score

count = 0
for i in range(len(result)):
    if train["helpful"][i] == result[i]:
        count += 1
print count
print "Percent uncorrectly classified ", ( len(result) - count ) * 100 / len(result )
print "Percent correctly classified ", count * 100 / len( result )

gnb = GaussianNB()
scores = cross_val_score( gnb, train_centroids, train["helpful"], cv=10 )
print "Scores - \n",  scores
print scores.mean()
