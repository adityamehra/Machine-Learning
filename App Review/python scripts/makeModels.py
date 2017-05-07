import gensim
import gensim.models.word2vec as w2v
import logging
import multiprocessing
import nltk
from nltk.tokenize import RegexpTokenizer
from nltk import pos_tag
import os
from os import listdir
import string
import sklearn.manifold
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

# Making the tokenizer to select only the alphanumeric characters.
# http://stackoverflow.com/questions/15547409/how-to-get-rid-of-punctuation-using-nltk-tokenizer
tokenizer = RegexpTokenizer(r'\w+')


dataPath  = "../processed data/"

# get all the files from the data folder
reviewFiles = listdir(dataPath)

def getSentences(reviewFile):
    sentences = []
    with open(dataPath+reviewFile) as processedFile:
        for line in processedFile:
            tokens = tokenizer.tokenize(line)
            if len(tokens) > 0:
                sentences.append(tokens)
    return sentences

# This function makes Word2Vec models and returns the model...
def makeModel(sentences):

    #3 main tasks that vectors help with
    #DISTANCE, SIMILARITY, RANKING

    # Dimensionality of the resulting word vectors.
    # more dimensions, more computationally expensive to train
    # but also more accurate
    # more dimensions = more generalized
    num_features = 200

    # Minimum word count threshold.
    min_word_count = 3

    # Number of threads to run in parallel.
    # more workers, faster we train
    num_workers = multiprocessing.cpu_count()

    # Context window length.
    context_size = 4

    # Downsample setting for frequent words.
    # 0 - 1e-5 is good for this
    downsampling = 1e-3

    # Seed for the RNG, to make the results reproducible.
    # random number generator
    # deterministic, good for debugging
    seed = 1

    reviews2vec = w2v.Word2Vec(
        sg=1,
        seed=seed,
        workers=num_workers,
        size=num_features,
        min_count=min_word_count,
        window=context_size,
        #sample=downsampling
    )
    reviews2vec.build_vocab( sentences )
    reviews2vec.train( sentences, total_examples = reviews2vec.corpus_count, epochs = reviews2vec.iter )
    return reviews2vec

def saveModel(model, modelName):
    modelName = modelName[:-4]
    if not os.path.exists("../trained-models3/"):
        os.mkdir("../trained-models3/")
    model.save("../trained-models3/" + modelName + ".w2v")

def makeModels():
    for reviewFile in reviewFiles:
        # making sure that only text files are processed
        if reviewFile.endswith(".txt"):
        # if reviewFile.endswith("allReviews.txt"):
            sentences = getSentences(reviewFile)
            model = makeModel(sentences)
            saveModel(model, reviewFile)


def getPoints():
    tsne = sklearn.manifold.TSNE( n_components =  2, random_state = 0 )
    all_word_vectors_matrix = reviews2vec.wv.syn0
    all_word_vectors_matrix_2d = tsne.fit_transform(all_word_vectors_matrix)
    points = pd.DataFrame(
        [
            (word, coords[0], coords[1])
            for word, coords in [
                (word, all_word_vectors_matrix_2d[reviews2vec.wv.vocab[word].index])
                for word in reviews2vec.wv.vocab
            ]
        ],
    columns=["word", "x", "y"]
    )
    return points

def plotModel():
    points = getPoints()
    plt.figure()
    points.plot.scatter("x", "y")
    plt.savefig("../plots/facebook_plot1.pdf")

def plot_region(x_bounds, y_bounds):
    points = getPoints()
    slice = points[
        (x_bounds[0] <= points.x) &
        (points.x <= x_bounds[1]) &
        (y_bounds[0] <= points.y) &
        (points.y <= y_bounds[1])
    ]
    plt.figure()
    ax = slice.plot.scatter("x", "y")
    for i, point in slice.iterrows():
        ax.text(point.x + 0.005, point.y + 0.005, point.word, fontsize=11)
    plt.savefig("../plots/facebook_plot2.pdf")

# reviews2vec = makeModel()
# reviews2vec = w2v.Word2Vec.load("../trained-models/facebook.w2v")
# print len(reviews2vec.wv.vocab)
# print reviews2vec.most_similar("facebook")
# plotModel()
# plot_region(x_bounds=(-50.0, 50.0), y_bounds=(-50.0, 50.0))

makeModels()
