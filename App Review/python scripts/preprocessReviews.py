import logging
import multiprocessing
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from nltk import pos_tag
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer
import os
from os import listdir
import pickle
import string
import sklearn.manifold
import numpy as np

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

stemmer = PorterStemmer()
lemmatiser = WordNetLemmatizer()

# Making the tokenizer to select only the alphanumeric characters.
# http://stackoverflow.com/questions/15547409/how-to-get-rid-of-punctuation-using-nltk-tokenizer
tokenizer = RegexpTokenizer(r'\w+')

# Generate a list of alphabets
# http://stackoverflow.com/questions/16060899/alphabet-range-python
# Convert a list to set in python
#http://stackoverflow.com/questions/15768757/how-to-construct-a-set-out-of-list-items-in-python
setOfNumbers = set(list("0123456789"))
setOfLetters = set(list(string.ascii_lowercase))
myStopWords = set(('hi', 'im', 'hey', 'hello', 'sooooo',  'sooo', 'aahh', 'th', 'th7', 'th8', 'th9', 'th10', 'th11', 'lol', 'us', 'si', 'shi', 'quot'))

# http://stackoverflow.com/questions/19130512/stopword-removal-with-nltk
# getting the stop words in english from nltk
# cahing the stopwords object for faster result
# http://stackoverflow.com/questions/19560498/faster-way-to-remove-stop-words-in-python
stopWords = set(stopwords.words("english")) | myStopWords | setOfLetters | setOfNumbers

inputPath = "../data/"
outputPath  = "../processed data/"

if not os.path.exists(outputPath):
    os.mkdir(outputPath)

# get all the files from the data folder
reviewFiles = listdir(inputPath)

# This function preprocesses the reviews -
# 1. Changes words to lowercase
# 2. Removes stop words (words which have no semantic meaning)
# 3. Stemms words to verb form
def preprocess():
    for reviewFile in reviewFiles:
        # making sure that only text files are processed
        if reviewFile.endswith(".txt"):
            outFile = open(outputPath + reviewFile, "w")
            # outFile = open(outputPath + "allReviews.txt", "a")
            with open(inputPath+reviewFile) as inputFile:
                for line in inputFile:
                    # http://stackoverflow.com/questions/12468179/unicodedecodeerror-utf8-codec-cant-decode-byte-0x9c
                    # line.lower() changes everything to lower case
                    line = unicode(line.lower(), errors='ignore')
                    # gets all the alphanumeric tokens as a list
                    tokens = tokenizer.tokenize(line)
                    # removes the stop words from the tokens
                    filtered_tokens = [token for token in tokens if token not in stopWords]
                    # gets the stemmed version of the filtered tokens
                    stemmed_tokens = [lemmatiser.lemmatize(token, pos="v") for token in filtered_tokens]
                    # outFile.write(' '.join(tokens) + "\n")
                    # outFile.write(' '.join(filtered_tokens) + "\n")
                    # print stemmed_tokens
                    if len(stemmed_tokens) > 0:
                        outFile.write(' '.join(stemmed_tokens) + "\n")

preprocess()
