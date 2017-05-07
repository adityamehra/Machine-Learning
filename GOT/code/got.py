import gensim
import gensim.models.word2vec as w2v
import logging
import multiprocessing
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from nltk import pos_tag
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import os
from os import listdir
import pickle
import string
import sklearn.manifold
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

lemmatiser = WordNetLemmatizer()

# Making the tokenizer to select only the alphanumeric characters.
# http://stackoverflow.com/questions/15547409/how-to-get-rid-of-punctuation-using-nltk-tokenizer
tokenizer = RegexpTokenizer(r'\w+')

# http://stackoverflow.com/questions/19130512/stopword-removal-with-nltk
# getting the stop words in english from nltk
# cahing the stopwords object for faster result
# http://stackoverflow.com/questions/19560498/faster-way-to-remove-stop-words-in-python
stopWords = set(stopwords.words("english"))

inputPath = "../data/"
outPath  = "../processed data/"

if not os.path.exists(outPath):
    os.mkdir(outPath)

textFiles = listdir(inputPath)

sentences = []

def preprocessing():
    for textFile in textFiles:
        # uses only text files
        if textFile.endswith(".txt"):
            outFile = open(outPath + textFile, "w")
            with open(inputPath+textFile) as inputFile:
                for line in inputFile:
                    # http://stackoverflow.com/questions/12468179/unicodedecodeerror-utf8-codec-cant-decode-byte-0x9c
                    # line.lower() changes everything to lower case
                    line = unicode(line.lower(), errors='ignore')
                    # gets all the alphanumeric tokens as a list
                    tokens = tokenizer.tokenize(line)
                    # removes the stop words from the tokens
                    filtered_tokens = [token for token in tokens if token not in stopWords]
                    if len(stemmed_tokens) > 0:
                        sentences.append(filtered_tokens)
                        outFile.write(' '.join(filtered_tokens) + "\n")

 # This function makes Word2Vec models and returns the model...
def makeModel():

    #3 main tasks that vectors help with
    #DISTANCE, SIMILARITY, RANKING

    # Dimensionality of the resulting word vectors.
    # more dimensions, more computationally expensive to train
    # but also more accurate
    # more dimensions = more generalized
    num_features = 300

    # Minimum word count threshold.
    min_word_count = 3

    # Number of threads to run in parallel.
    # more workers, faster we train
    num_workers = multiprocessing.cpu_count()

    # Context window length.
    context_size = 7

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
        sample=downsampling
    )

    reviews2vec.build_vocab(sentences)

    reviews2vec.train(sentences, total_examples=reviews2vec.corpus_count, epochs=reviews2vec.iter)

    if not os.path.exists("../trained-models/"):
        os.mkdir("../trained-models/")

    reviews2vec.save("../trained-models/got.w2v")

    return reviews2vec

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
    #print points
    plt.figure()
    points.plot.scatter("x", "y", s=10, figsize=(20, 12))
    plt.savefig("../plots/got_plot1.pdf")

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
    plt.savefig("../plots/got_plot2.pdf")

def nearest_similarity_cosmul(start1, end1, end2):
    similarities = reviews2vec.most_similar_cosmul(
        positive=[end2, start1],
        negative=[end1]
    )
    start2 = similarities[0][0]
    print("{start1} is related to {end1}, as {start2} is related to {end2}".format(**locals()))
    print similarities
    print "\n"
    # return start2


#preprocessing()
#reviews2vec = makeModel()
reviews2vec = w2v.Word2Vec.load("../trained-models/got.w2v")
print len(reviews2vec.wv.vocab)
#print reviews2vec.most_similar("stark")
#print reviews2vec.most_similar("jon")
#plotModel()
#plot_region(x_bounds=(4.0, 4.2), y_bounds=(-0.5, -0.1))
nearest_similarity_cosmul("stark", "winterfell", "riverrun")
nearest_similarity_cosmul("robb", "stark", "lannister")
nearest_similarity_cosmul("jaime", "sword", "wine")
nearest_similarity_cosmul("arya", "nymeria", "dragons")
nearest_similarity_cosmul("jon", "snow", "tully")
nearest_similarity_cosmul("jon", "snow", "tarly")
nearest_similarity_cosmul("arya", "stark", "tarly")
nearest_similarity_cosmul("bran", "robb", "tyrion")
nearest_similarity_cosmul("bran", "robb", "jaime")
nearest_similarity_cosmul("eddard", "robb", "tyrion")
nearest_similarity_cosmul("eddard", "robb", "jaime")
nearest_similarity_cosmul("eddard", "robb", "tywin")
nearest_similarity_cosmul("sansa", "robb", "jaime")
nearest_similarity_cosmul("arya", "robb", "jaime")
nearest_similarity_cosmul("robb", "sansa", "cersei")
nearest_similarity_cosmul("cersei", "robert", "eddard")
nearest_similarity_cosmul("cersei", "tywin", "eddard")
nearest_similarity_cosmul("cersei", "jaime", "tyrion")
