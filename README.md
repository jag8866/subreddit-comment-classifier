# Subreddit Comment Classifier

Allows classification of raw comment text into one of two source subreddits, using three different classification methods: 
Support Vector Machines, Random Forests, and neural networks (specifically LSTMs). Also uses two different word embedding
methods: sklearn's HashingVectorizer and Google's Word2Vec (gensim implementation).

### Prerequisites

* Python 2.7 (Anaconda)
* keras with tensorflow backend
* nltk
* gensim (Word2Vec embeddings)
* sklearn (SVM, Random Forest)
* praw (for pulling reddit comments)

## Usage

Before anything else you will have to create a Reddit application at ```https://www.reddit.com/prefs/apps``` 
and put the corresponding information into the authentication arguments of praw.Reddit() at the top of 
```redditharvest.py```.

```python svm_and_rf.py sub1 sub2``` downloads a corpus of comments from subreddits named sub1 and sub2, 
processes them, trains an svm and random forest on them, then runs tests on both and prints their 
results. For example: ```python svm_and_rf.py politics funny``` This uses sklearn's 
HashingVectorizer to create word vectors.

```python lstm.py sub1 sub2``` downloads a corpus of comments from subreddits named sub1 and sub2, processes 
them, trains an lstm on them, then runs tests and prints results. For example: ```python lstm.py politics funny```
This uses the gensim implementation of Google's Word2Vec to create word vectors.

## Results

To test these classifiers I chose to use the controversial pair of subreddits "theredpill" and "thebluepill". The
reason these were chosen is because I feel that they would be a fair test of the classifier's abilities - since 
”thebluepill” is a subreddit about ”theredpill” there is a lot of crossover in terminology so classifying a 
comment by key words alone will hopefully not be as effective a strategy. Though this makes it more difficult, 
even a little understanding of tone and intent should go a long way since the users of these two subreddits 
differ drastically in their views. Over twelve thousand comments were downloaded to use as data. 

The lists of examples from each subreddit are truncated down to the length of the shortest one to ensure that
there is an exact 50/50 split between them. This is important because otherwise some models will end up always
choosing whichever one is more common, resulting in a bad classifier that may appear more successful than it is
because its accuracy will converge towards the percentage of examples from the most common sub. Since we know
we have split it 50/50, this means the lowest accuracy we should expect to see is 50%. Anything above 50% should
indicate some level of success.

With this in mind, my results are as follows: the Random Forest scored the lowest at around 62% accuracy, the SVM 
was a bit better at 64%, and the LSTM scored about 69%. This is pretty much in line with expectations - LSTMs have been
shown to perform extremely well on natural language classification tasks and usually outperform older methods like
SVMs and random forests. Some of this LSTM's success may also be attributed to the Word2Vec embeddings that it uses,
which have also been shown to generally be more effective for tasks like these than other methods.

It is worth noting that the LSTM suffered from severe overfitting in my tests. Some possible next steps to improve 
results and fight overfitting might be: tuning LSTM hyperparameters (currently it is not a deep LSTM, 2 or 3 layers 
may be more effective than one) and using more data.
