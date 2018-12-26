'''
Author - Joseph Golden
Collects comments from two subreddits, converts words to word2vec vectors,
and trains an LSTM to predict which sub a comment is from, then tests
them and prints accuracy ratings and some examples of
comments and their predicted/actual sources.
'''

from gensim.models import Word2Vec
import nltk
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.ensemble import RandomForestRegressor
import re
import pickle
from sklearn import svm
import numpy
from keras.models import Sequential
import keras.models
from keras.layers import Dense
from keras.layers import CuDNNLSTM
from keras.layers import TimeDistributed
from keras.layers import Dropout
from keras.preprocessing.sequence import pad_sequences
from pathlib import Path
import matplotlib.pyplot as plt
import sys

if len(sys.argv) == 3:
	sub1_name = sys.argv[1]
	sub2_name = sys.argv[2]
	sub1 = sys.argv[1]
	sub2 = sys.argv[2]
else:
	raise Exception("usage python lstm.py sub1 sub2")

if not (Path(sub1 + ".pk2").is_file() and Path(sub2 + ".pk2").is_file()):
	print("Collecting comments for " + sub1 + "\n")
	sub1 = get_comments(sub1)
	print("Collecting comments for " + sub2 + "\n")
	sub2 = get_comments(sub2)
else:
	#Pull subreddit comments from files
	with open(sub1 + ".pk2", "rb") as f:
		red = pickle.load(f)

	with open(sub2 + ".pk2", "rb") as f:
		blue = pickle.load(f)

#Preprocess (lowercase)
temp = []
for line in red:
	newline = line
	newline = newline.lower()
	temp += [newline]
red = list(temp)

temp = []
for line in blue:
	newline = line
	newline = newline.lower()
	temp += [newline]
blue = list(temp)

#Create a Word2Vec model unless one is already saved
w2vfile = Path("word2vec.model")
if not w2vfile.is_file():
	all_words = [nltk.word_tokenize(sent) for sent in blue + red]
	word2vec = Word2Vec(all_words, min_count=1)
	word2vec.save("word2vec.model")

else:
	word2vec = Word2Vec.load("word2vec.model")

#Turn comments into word2vec arrays X and Y unless theyre already saved
pick = Path("data.pk2")
if not pick.is_file():
	temp = []
	for line in red:
		newline = []
		for word in nltk.word_tokenize(line):
			newline += [word2vec.wv[word].tolist()]
		temp += [newline]
	red = numpy.array(temp)

	temp = []
	for line in blue:
		newline = []
		for word in nltk.word_tokenize(line):
			newline += [word2vec.wv[word].tolist()]
		temp += [newline]
	blue = numpy.array(temp)

	X, Y = [], []
	#Interleave the subreddit's data, use the length of the shortest so its 50/50
	for i in range(0, min(len(blue), len(red))):
		X.append(blue[i])
		Y.append(0)
		X.append(red[i])
		Y.append(1)
	pickle.dump([X, Y], open("data.pk2", "wb"))

else:
	l = pickle.load(open("data.pk2", "rb"))
	X = l[0]
	Y = l[1]

#Split into train, test, and development sets
X_train = X[:len(X)//2]
X_dev = X[len(X)//2:(3 * len(X))//4]
X_test = X[(3 * len(X))//4:]

Y_train = numpy.array(Y[:len(Y)//2])
Y_dev = numpy.array(Y[len(Y)//2:(3 * len(Y))//4])
Y_test = numpy.array(Y[(3 * len(Y))//4:])

X_train = pad_sequences(X_train, dtype='float', maxlen=128)
X_dev = pad_sequences(X_dev, dtype='float', maxlen=128)
X_test = pad_sequences(X_test, dtype='float', maxlen=128)

X_train = X_train.reshape(-1, 128, 100)
X_dev = X_dev.reshape(-1, 128, 100)

#Create neural net model and train it, unless a trained model is already saved
nnpath = Path("classifier.h5")
if not nnpath.is_file():
	model = Sequential()
	model.add(CuDNNLSTM(32, input_shape=(128, 100)))
	model.add(Dropout(.3))
	model.add(Dense(1, activation='sigmoid'))

	model.compile(loss='mean_squared_error', optimizer='adam')
	history = model.fit(X_train, Y_train, validation_data= (X_dev, Y_dev), epochs=100, verbose=2, batch_size=50)
	model.save("classifier.h5")
	print(model.summary())

	# Plotting the training and validation stats to help detect overfitting and other issues
	loss = history.history['loss']
	val_loss = history.history['val_loss']
	epochs = range(len(loss))
	plt.figure()
	plt.plot(epochs, loss, 'bo', label='Training loss')
	plt.plot(epochs, val_loss, 'b', label='Validation loss')
	plt.title("Training and Validation Loss")
	plt.legend()
	plt.show()


else:
	model = keras.models.load_model("classifier.h5")
	print(model.summary())


#Performance measurements for the neural net
correct = 0
incorrect = 0
for i in range(0, len(X_test) - 1):
	predict = numpy.round(model.predict([X_test[i:i+1]]))

	if predict == Y_test[i]:
		correct += 1
	else:
		incorrect += 1
	if i % 400 == 0 and correct != 0 and incorrect != 0:
		print("Accuracy: " + str(100 * (float(correct) / (correct + incorrect))) )
		print(re.sub(r'^((\.) )*', '', " ".join([str(word2vec.similar_by_vector(X_test[i][x], topn=1)[0][0].encode('utf8')) for x in range(len(X_test[i]))])))
		print("Predicted: " + (sub2_name if predict == 0 else sub1_name) + ". Correct answer was: " + (sub2_name if Y_test[i] == 0 else sub1_name))
