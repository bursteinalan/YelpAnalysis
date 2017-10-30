import urllib.request
import json
import dml
import prov.model
import datetime
import uuid
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score

from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix



# read the data from disk and split into lines
# we use .strip() to remove the final (empty) line

def getData():
	client = dml.pymongo.MongoClient()
	repo = client.repo
	repo.authenticate('alanbur', 'alanbur')
	collection=repo['yelp.reviews'].find();
	reviews=[review for review in collection]
	return reviews


def balance_classes(xs, ys):
	"""Undersample xs, ys to balance classes."""
	freqs = Counter(ys)

	# the least common class is the maximum number we want for all classes
	max_allowable = freqs.most_common()[-1][1]
	num_added = {clss: 0 for clss in freqs.keys()}
	new_ys = []
	new_xs = []
	for i, y in enumerate(ys):
		if num_added[y] < max_allowable:
			new_ys.append(y)
			new_xs.append(xs[i])
			num_added[y] += 1
	return new_xs, new_ys

reviews=getData()
print(len(reviews))
# we're interested in the text of each review 
# and the stars rating, so we load these into 
# separate lists
texts = [review['text'] for review in reviews]
stars = [review['stars'] for review in reviews]



print(Counter(stars))
balanced_x, balanced_y = balance_classes(texts, stars)
print(Counter(balanced_y))


# This vectorizer breaks text into single words and bi-grams
# and then calculates the TF-IDF representation
vectorizer = TfidfVectorizer(ngram_range=(1,2))
t1 = datetime.datetime.now(tz=None)

# the 'fit' builds up the vocabulary from all the reviews
# while the 'transform' step turns each indivdual text into
# a matrix of numbers.
vectors = vectorizer.fit_transform(balanced_x)
print(datetime.datetime.now(tz=None) - t1)

X_train, X_test, y_train, y_test = train_test_split(vectors, balanced_y, test_size=0.33, random_state=42)


# initialise the SVM classifier
classifier = LinearSVC()
 
# train the classifier
t1 = datetime.datetime.now()
classifier.fit(X_train, y_train)
print(datetime.datetime.now() - t1)

preds = classifier.predict(X_test)
print(list(preds[:10]))
print(y_test[:10])

# print(texts[0])
# print(texts[4])

print(accuracy_score(y_test, preds))

print(classification_report(y_test, preds))

print(confusion_matrix(y_test, preds))



