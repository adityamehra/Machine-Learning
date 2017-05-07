import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import gensim.models.word2vec as w2v
import logging
import multiprocessing
import os

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

train = []

# print train.shape
# print train["review"][6917]

def getData():
    train = pd.read_csv("../data/training.txt", sep = "\t", header = None)
    train.columns = ["sentiment", "review"]
    reviews = []
    stopWords = set(stopwords.words("english"))
    for review in train["review"]:
        tokens = word_tokenize(review)
        tokens = [token for token in tokens if token not in stopWords]
        if len(tokens) > 0:
            reviews.append(tokens)
    return reviews

def makeModel(sentences):
    # 3 main tasks that vectors help with
    # DISTANCE, SIMILARITY, RANKING
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
    context_size = 4

    # Downsample setting for frequent words.
    # 0 - 1e-5 is good for this
    downsampling = 1e-3

    # Seed for the RNG, to make the results reproducible.
    # random number generator
    # deterministic, good for debugging
    seed = 1

    model = w2v.Word2Vec(
        sg=1,
        seed=seed,
        workers=num_workers,
        size=num_features,
        min_count=min_word_count,
        window=context_size,
        #sample=downsampling
    )
    model.build_vocab( sentences )
    model.train( sentences, total_examples = model.corpus_count, epochs = model.iter )
    return model

def saveModel(model, modelName):
    if not os.path.exists("../trained-models/"):
        os.mkdir("../trained-models/")
    model.save("../trained-models/" + modelName + ".w2v")

reviews = getData()
model = makeModel(reviews)
saveModel(model, "model")
# print reviews
model = w2v.Word2Vec.load("../trained-models/model.w2v")

# print "\potter"
# print model.most_similar("potter")
