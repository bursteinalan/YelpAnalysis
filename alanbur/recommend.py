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

import random
import math

class YelpSVM():
	Allreviews=[]
	def getXPercent(self,percent):
		'''
		From sample get n percent of the data
		Use Reservoir sampling
		'''
		# client = dml.pymongo.MongoClient()
		# repo = client.repo
		# repo.authenticate('alanbur', 'alanbur')

		

		# collection=repo['yelp.reviews'].find()
		# Allreviews=[review for review in collection]
		
		# print(kSize)
		print('randomly sampling')
		kSize=math.floor(len(self.Allreviews)*percent)
		reviews=self.Allreviews[:kSize]
		for i in range(kSize+1,len(self.Allreviews)):
			j=random.randint(1,i)
			if j<kSize:
				reviews[j] = self.Allreviews[i]
		# reviews=[review for review in collection]
		return reviews


	def getData(self):
		'''
		Reads from Mongo into mememory
		'''
		print('Getting all reviews')
		client = dml.pymongo.MongoClient()
		repo = client.repo
		repo.authenticate('alanbur', 'alanbur')
		collection=repo['yelp.reviews'].find();
		reviews=[review for review in collection]
		self.Allreviews=reviews
		return reviews


	def balance_classes(self,xs, ys):
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

	def run(self,percent, showDetails):
		'''
		runs on "percent" of data
		'''
		# reviews=getData()
		reviews=self.getXPercent(percent)
		print(len(reviews),' reviews to analyze')
		# we're interested in the text of each review 
		# and the stars rating, so we load these into 
		# separate lists
		texts = [review['text'] for review in reviews]
		stars = [review['stars'] for review in reviews]



		print(Counter(stars))
		balanced_x, balanced_y = self.balance_classes(texts, stars)
		print(Counter(balanced_y))

		unbalanced_x,unbalanced_y = texts, stars
		print(Counter(balanced_y))

		# This vectorizer breaks text into single words and bi-grams
		# and then calculates the TF-IDF representation
		vectorizer = TfidfVectorizer(ngram_range=(1,2))
		t1 = datetime.datetime.now(tz=None)

		# print(vectorizer)

		# the 'fit' builds up the vocabulary from all the reviews
		# while the 'transform' step turns each indivdual text into
		# a matrix of numbers.
		vectors = vectorizer.fit_transform(balanced_x)
		vectors2 = vectorizer.fit_transform(unbalanced_x)

		# print(vectors)
		if showDetails:
			print('form to classifier time: ',datetime.datetime.now(tz=None) - t1)

		X_train, X_test, y_train, y_test = train_test_split(vectors, balanced_y, test_size=0.33, random_state=42)
		X_train2, X_test2, y_train2, y_test2 = train_test_split(vectors2, unbalanced_y, test_size=0.33, random_state=42)


		# initialise the SVM classifier
		classifier = LinearSVC()
		classifier2= LinearSVC()
		# train the classifier
		t1 = datetime.datetime.now()
		classifier.fit(X_train, y_train)
		classifier2.fit(X_train2, y_train2)
		if showDetails:
			print('Train time: ',datetime.datetime.now() - t1)

		preds = classifier.predict(X_test)
		preds2 = classifier2.predict(X_test2)
		if showDetails:
			print(list(preds[:10]))
			print(y_test[:10])

		# print(texts[0])
		# print(texts[4])

		print('Balanced Accuracy with ', percent, ' of the data is: ',accuracy_score(y_test, preds))
		print('UnBalanced Accuracy with ', percent, ' of the data is: ',accuracy_score(y_test2, preds2))
		if showDetails:
			print(classification_report(y_test, preds))
			print(confusion_matrix(y_test, preds))

YSVM=YelpSVM()
YSVM.getData()
YSVM.run(1, False)
# YSVM.run(.5, False)
# YSVM.run(.25, False)
# YSVM.run(.125, False)
# YSVM.run(.0625, False)
# YSVM.run(.01, False)


# print(len(getXPercent(.01)))

