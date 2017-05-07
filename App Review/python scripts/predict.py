import pandas as pd
from sklearn import svm
from sklearn.model_selection import cross_val_score

data_train = pd.read_csv("train.csv", header=0, delimiter=",")
print data_train.shape
print data_train["length"].shape
print data_train[["length"]].shape

print data_train["length"]
print data_train[["length"]]

X_train = data_train[["length", "rating"]]
print X_train.shape

clf = svm.SVC()

scores = cross_val_score( clf, X_train, data_train["helpful"], cv=10 )
print "Scores 1 - \n",  scores
print scores.mean()

X_train = data_train[["length"]]
scores = cross_val_score( clf, X_train, data_train["helpful"], cv=10 )
print "Scores 2 - \n",  scores
print scores.mean()

X_train = data_train[["rating"]]
scores = cross_val_score( clf, X_train, data_train["helpful"], cv=10 )
print "Scores 3- \n",  scores
print scores.mean()
