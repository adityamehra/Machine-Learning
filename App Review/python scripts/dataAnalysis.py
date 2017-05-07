import os
from os import listdir
import re
import pandas as pd
from nltk.tokenize import word_tokenize
import numpy as np
from random import randint
# RUNNAME = "facebook"
RUNNAME = "clashofclans"
# RUNNAME = "allReviews"
# RUNNAME = "gmail"
# RUNNAME = "instagram"

print RUNNAME

train = pd.read_csv("../processed data/" + RUNNAME + ".csv", header=0, delimiter=",", quoting=2)

print train.shape
print train.columns

count_0 = 0
count_1 = 0
print len(train["helpful"])
for i in range( len( train["helpful"] ) ):
    if train["helpful"][i] == 1:
        count_1 += 1
    elif train["helpful"][i] == 0:
        count_0 += 1

print "Number of zero", count_0
print "Number of one", count_1

reviewLength = np.empty(shape = (0));


for review in train["review"]:
    tokens = word_tokenize(review)
    reviewLength = np.append(reviewLength, len(tokens))

train["length"] = reviewLength.T
# print train.shape

# print train

reviewRating = np.empty(shape = (0))

for length in train["length"]:
    if length > 10:
        reviewRating = np.append(reviewRating, randint(1, 3))
    else:
        reviewRating = np.append(reviewRating, randint(4, 5))

train["rating"] = reviewRating.T
print train.shape

# print train[["length", "rating"]]

output = pd.DataFrame( train )
output.to_csv( "train.csv", index= False)
