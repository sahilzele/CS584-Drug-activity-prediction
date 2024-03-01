#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 27 11:15:05 2023

@author: sahil
"""

import re
from collections import Counter
from sklearn.model_selection import KFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from imblearn.over_sampling import SMOTE
from sklearn.naive_bayes import BernoulliNB
from sklearn.model_selection import cross_val_score
from sklearn.metrics import make_scorer, f1_score
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
########################################################################
train_file = "/Users/sahil/Documents/CS584/Ass2/train.txt"
test_file = "/Users/sahil/Documents/CS584/Ass2/test.txt"

#declaring upper bound for no of features
no_of_features = 100001

########################################################################
# load file method to update 1 where the feature is present  and 0 otherwise
def loadFile(filename, filetype):
    with open(filename, "r") as readfile:
        lines = readfile.readlines()

    if filetype == "train":
        labels = [int(l[0]) for l in lines]
        for index, item in enumerate(labels):
            if (item == 0):
                labels[index] = 0
        docs = [re.sub(r'[^\w]', ' ',l[1:]).split() for l in lines]

    else:
        labels = []
        docs = [re.sub(r'[^\w]', ' ',l).split() for l in lines]

    features = []

    for doc in docs:
        line = [0]*no_of_features
        for index, val in enumerate(doc):
            line[int(val)] = 1
        features.append(line)

    return features, labels

#loading train dataset using loadfile method
features, labels = loadFile(train_file, "train")

#loading test dataset using loadfile method
test_features, test_labels = loadFile(test_file, "test")
#########################################################################
#using Selectkbest for feature selection
k = 150

# Initialized the SelectKBest
selector = SelectKBest(score_func=f_classif, k=k)

# Fit the feature selector to the training data and transform it
reduced_features = selector.fit_transform(features, labels)
test_reduced_features = selector.transform(test_features)
#########################################################################
#counter to check the label value count
'''
counter = Counter(labels)
print(counter)

#using SMOTE for oversampling
sm = SMOTE(random_state=42)
reduced_features, labels = sm.fit_resample(features, labels)
counter = Counter(labels)
print(counter)
'''

#########################################################################
# using decision tree classifier and Naive bayes model(GaussianNB and BernoulliNB model)
dt_classifier = DecisionTreeClassifier()
nb_classifier = GaussianNB()
bnb_model = BernoulliNB()


# fitting all three models and camparing their F1_score
bnb_model.fit(reduced_features, labels)
dt_classifier.fit(reduced_features, labels)
nb_classifier.fit(reduced_features, labels)
# Define the scoring function for cross-validation
scorer = make_scorer(f1_score, average='weighted')

bnb_f1_scores = cross_val_score(bnb_model, reduced_features, labels, cv=KFold(n_splits=5, shuffle=True, random_state=42), scoring=scorer)

dt_f1_scores = cross_val_score(nb_classifier, reduced_features, labels, cv=KFold(n_splits=5, shuffle=True, random_state=42), scoring=scorer)

dtc_f1_scores = cross_val_score(dt_classifier, reduced_features, labels, cv=KFold(n_splits=5, shuffle=True, random_state=42), scoring=scorer)


#comparing F1_score
print(dtc_f1_scores.mean())

print(bnb_f1_scores.mean())

print(dt_f1_scores.mean())


#prediciting the output
Output =bnb_model.predict(test_reduced_features)

#Output file creation 
file_path = '/Users/sahil/Documents/CS584/Ass2/format.dat'
# Open the file in write mode ('w')
with open(file_path, 'w') as file:
    # Write each element of the list to the file
    for item in Output:
        file.write(f'{item}\n')