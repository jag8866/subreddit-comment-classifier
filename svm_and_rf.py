'''
Author - Joseph Golden
Collects comments from two subreddits, turns the words into hash vectors,
and trains SVM and Random Forest classifiers to predict which sub a comment
is from, then tests them and prints accuracy ratings and some examples of
comments and their predicted/actual sources.
'''

import nltk
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.ensemble import RandomForestRegressor
import re
import pickle
from sklearn import svm
import numpy
from redditharvest import get_comments
import sys
from pathlib import Path

#Arguments are the names of the subs
if len(sys.argv) == 3:
	sub1_name = sys.argv[1]
	sub2_name = sys.argv[2]
else:
	raise Exception("usage python svm_and_rf.py sub1 sub2")


if not (Path(sub1_name + ".pk2").is_file() and Path(sub2_name + ".pk2").is_file()):
	print("Collecting comments for " + sub1_name + "\n")
	sub1 = get_comments(sub1_name)
	print("Collecting comments for " + sub2_name + "\n")
	sub2 = get_comments(sub2_name)
else:
	#Pull subreddit comments from files
	with open(sub1_name + ".pk2", "rb") as f:
		sub1 = pickle.load(f)

	with open(sub2_name + ".pk2", "rb") as f:
		sub2 = pickle.load(f)

#Preprocess (lowercase)
temp = []
for line in sub1:
	newline = line
	newline = newline.lower()
	temp += [newline]
sub1 = temp[:]

temp = []
for line in sub2:
	newline = line
	newline = newline.lower()
	temp += [newline]
sub2 = temp[:]

#Interleave the subreddit's data into X and create Y, use the length of the shortest set so its 50/50
X_raw, Y = [], []
for i in range(0, min(len(sub2), len(sub1))):
	X_raw.append(sub2[i])
	Y.append(0)
	X_raw.append(sub1[i])
	Y.append(1)



#Create hash table of words
vectorizer = HashingVectorizer(n_features=100)
vector = vectorizer.transform(X_raw)
X = vector.toarray()
hash_lookup = {str(X[i]): X_raw[i] for i in range(len(X))}


#Split into train, test, and development sets
X_train = X[:len(X)//2]
X_dev = X[len(X)//2:(3 * len(X))//4]
X_test = X[(3 * len(X))//4:]

Y_train = Y[:len(Y)//2]
Y_dev = Y[len(Y)//2:(3 * len(Y))//4]
Y_test = Y[(3 * len(Y))//4:]

#SVM creation and training
svm_classifier = svm.SVC(kernel='linear', gamma='scale')
svm_classifier.fit(X_train, Y_train)

#SVM performance measurements
correct = 0
incorrect = 0
for i in range(len(X_test)):
	predict = svm_classifier.predict([X_test[i]])

	if predict == Y_test[i]:
		correct += 1
	else:
		incorrect += 1
	if i % 1000 == 0 and correct != 0 and incorrect != 0:
		print("Accuracy: " + str(100 * (float(correct)/(correct + incorrect))))
		print(str(hash_lookup[str(X_test[i])]))
		print("Predicted: " + (sub2_name if predict == 0 else sub1_name) + ". Correct answer was: " + (sub2_name if Y_test[i] == 0 else sub1_name))
		

#Random Forest creation and training
rf_classifier = RandomForestRegressor(n_estimators = 100, random_state = 69)
rf_classifier.fit(X_train, Y_train);

#Random forest performance measurements
correct = 0
incorrect = 0
for i in range(len(X_test)):
	predict = rf_classifier.predict([X_test[i]])
	predict = numpy.round(predict)

	if predict == Y_test[i]:
		correct += 1
	else:
		incorrect += 1
	if i % 400 == 0 and correct != 0 and incorrect != 0:
		print("Accuracy: " + str(100 * (float(correct)/(correct + incorrect))))
		print(str(hash_lookup[str(X_test[i])]))
		print("Predicted: " + (sub2_name if predict == 0 else sub1_name) + ". Correct answer was: " + (sub2_name if Y_test[i] == 0 else sub1_name))
