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
import sys


liveDataStream = False

class YelpSVM():
	Allreviews=[]
	testReviews=[]
	# mostWordReviews=[]
	# leastWordReviews=[]
	reviewSubset=[]

	num = random.randrange(sys.maxsize)
	random.seed(num)
	print("Seed was:", num)

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
		print('ReAdd old test data')
		self.Allreviews=self.Allreviews+self.reviewSubset
		self.reviewSubset=[]
		# self.Allreviews=self.Allreviews+self.mostWordReviews
		# self.mostWordReviews=[]
		print('randomly sampling')
		kSize=math.floor(len(self.Allreviews)*percent)
		reviews=self.Allreviews[:kSize]
		for i in range(kSize+1,len(self.Allreviews)):
			j= random.randint(1,i)
			if j<kSize:
				reviews[j] = self.Allreviews[i]
		# reviews=[review for review in collection]
		return reviews

	def testData(self, n):
		'''
		Get a number of documents from dataset for testing
		'''
		print('ReAdd old test data')
		# print('length of old test data: ',len(self.testReviews))
		self.Allreviews=self.Allreviews+self.testReviews
		self.testReviews=[]

		print('Building test data')
		print('Length of allreviews: ', len(self.Allreviews))
		for i in range(n):
			index=random.randint(0,len(self.Allreviews)-1)
			element=self.Allreviews.pop(index)
			self.testReviews.append(element)
		print('New length of allreviews: ', len(self.Allreviews))
		
	def getLeastWords(self,percent):
		'''
		Build reviews with Most words
		'''
		print('ReAdd old test data')
		# print('length of old test data: ',len(self.testReviews))
		self.Allreviews=self.Allreviews+self.reviewSubset
		self.reviewSubset=[]
		# self.Allreviews=self.Allreviews+self.mostWordReviews
		# self.mostWordReviews=[]

		
		print('Getting reviews with least words from allreviews: ', len(self.Allreviews))
		kSize=math.floor(len(self.Allreviews)*percent)
		print('Building lookup dictionary')
		
		wordcountDict=dict()
		for review in self.Allreviews:
			numWords=len(review['text'].split())
			if numWords in wordcountDict:
				wordcountDict[numWords].append(review)
				
			else:
				wordcountDict[numWords] = [review]

		print('Dictionary size: ', len(wordcountDict.keys()))
		
		print('Popping ',kSize, ' elements')
		self.reviewSubset=[]
		minKey=min(wordcountDict.keys())
		minDictElement=wordcountDict.pop(minKey)
		while kSize>0:
			if minDictElement==[]:
				minKey=min(wordcountDict.keys())
				minDictElement=wordcountDict.pop(minKey)
			minReview=minDictElement.pop()
			# self.Allreviews.remove(minReview)
			self.reviewSubset.append(minReview)
			kSize-=1
		
		#Faster implementation to update All reviews
		self.Allreviews=minDictElement
		for value in wordcountDict.values():
			self.Allreviews.extend(value)
		print('len of reviewSubset: ', len(self.reviewSubset))
		print('len of Shortest Review: ', len(self.reviewSubset[0]['text'].split()))
		print('len of Longest Review: ', len(self.reviewSubset[len(self.reviewSubset)-1]['text'].split()))

		
	def getMostWords(self,percent):
		'''
		Build reviews with Most words
		'''
		print('ReAdd old test data')
		# print('length of old test data: ',len(self.testReviews))
		self.Allreviews=self.Allreviews+self.reviewSubset
		self.reviewSubset=[]
		# self.Allreviews=self.Allreviews+self.mostWordReviews
		# self.mostWordReviews=[]

		print('Getting reviews with Most words from allreviews: ', len(self.Allreviews))
		kSize=math.floor(len(self.Allreviews)*percent)
		print('Building lookup dictionary')
		wordcountDict=dict()
		for review in self.Allreviews:
			numWords=len(review['text'].split())
			if numWords in wordcountDict:
				wordcountDict[numWords].append(review)
			else:
				wordcountDict[numWords] = [review]

		print('Dictionary size: ', len(wordcountDict.keys()))
		print('Popping ',kSize, ' elements')
		self.reviewSubset=[]
		maxKey=max(wordcountDict.keys())
		maxDictElement=wordcountDict.pop(maxKey)
		while kSize>0:
			if maxDictElement==[]:
				maxKey=max(wordcountDict.keys())
				maxDictElement=wordcountDict.pop(maxKey)
			maxReview=maxDictElement.pop()
			# self.Allreviews.remove(maxReview)
			self.reviewSubset.append(maxReview)
			kSize-=1
		self.Allreviews=maxDictElement
		for value in wordcountDict.values():
			self.Allreviews.extend(value)
		print('len of reviewSubset: ', len(self.reviewSubset))
		print('len of Allreviews: ', len(self.Allreviews))
		print('len of Shortest Review: ', len(self.reviewSubset[0]['text'].split()))
		print('len of Longest Review: ', len(self.reviewSubset[len(self.reviewSubset)-1]['text'].split()))


	def getWordLengthPercentile(self, percentile, percent):
		'''
		return the desired percent of data starting from a percentile
		'''
		'''
		Build reviews with Most words
		'''
		print('ReAdd old test data')
		# print('length of old test data: ',len(self.testReviews))
		self.Allreviews=self.Allreviews+self.reviewSubset
		self.reviewSubset=[]
		# self.Allreviews=self.Allreviews+self.mostWordReviews
		# self.mostWordReviews=[]
		# self.Allreviews=self.Allreviews+self.leastWordReviews
		# self.leastWordReviews=[]

		
		print('Number of reviews to choose from: ', len(self.Allreviews))
		print('Getting reviews starting from percent: ', percentile)

		kSize=math.floor(len(self.Allreviews)*percent)
		ignoreSize=math.floor(len(self.Allreviews)*percentile)
		print('Will get ',percent,' reviews which is: ', kSize)

		print('Building lookup dictionary')		
		wordcountDict=dict()
		for review in self.Allreviews:
			numWords=len(review['text'].split())
			if numWords in wordcountDict:
				wordcountDict[numWords].append(review)
				
			else:
				wordcountDict[numWords] = [review]

		print('Dictionary size: ', len(wordcountDict.keys()))
		
		self.Allreviews=[]

		print('ignoring ',ignoreSize, ' elements')
		minKey=min(wordcountDict.keys())
		minDictElement=wordcountDict.pop(minKey)
		while ignoreSize>0:
			if minDictElement==[]:
				minKey=min(wordcountDict.keys())
				minDictElement=wordcountDict.pop(minKey)
			minVal=minDictElement.pop()
			ignoreSize-=1
			self.Allreviews.append(minVal)
		self.Allreviews.extend(minDictElement)
		
		print('Popping ',kSize, ' elements')
		self.reviewSubset=[]
		minKey=min(wordcountDict.keys())
		minDictElement=wordcountDict.pop(minKey)
		while kSize>0 and len(wordcountDict)>0:
			if minDictElement==[]:
				minKey=min(wordcountDict.keys())
				minDictElement=wordcountDict.pop(minKey)
			minReview=minDictElement.pop()
			# self.Allreviews.remove(minReview)
			self.reviewSubset.append(minReview)
			kSize-=1
		
		#Faster implementation to update All reviews
		self.Allreviews.extend(minDictElement)
		for value in wordcountDict.values():
			self.Allreviews.extend(value)
		print('len of reviewSubset: ', len(self.reviewSubset))
		print('len of Shortest Review: ', len(self.reviewSubset[0]['text'].split()))
		print('len of Longest Review: ', len(self.reviewSubset[len(self.reviewSubset)-1]['text'].split()))

	def getData(self):
		'''
		Reads from Mongo into mememory
		'''
		reviews=[]
		print('Getting all reviews')
		if not liveDataStream:
			print('Getting from mongo')
			client = dml.pymongo.MongoClient()
			repo = client.repo
			repo.authenticate('alanbur', 'alanbur')
			collection=repo['yelp.reviews'].find();
			reviews=[review for review in collection]
		if liveDataStream:
			print('Getting from file')
			file='/Users/burstein/Documents/directedStudy/dataset/review.json'
			with open(file) as f:
				reviews = f.read().strip().split("\n")

				# each line of the file is a separate JSON object
				reviews = [json.loads(review) for review in reviews] 

		self.Allreviews=reviews
		# return reviews


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

	def run(self,percent, showDetails,whatToUse, percentile=0):
		'''
		runs on "percent" of data
		'''

		# reviews=getData()
		if whatToUse=='random':
			reviews=self.getXPercent(percent)
		elif whatToUse=='longestReviews':
			self.getMostWords(percent)
			reviews=self.reviewSubset
			print('len of reviewSubset: ', len(self.reviewSubset))
		elif whatToUse=='shortestReviews':
			self.getLeastWords(percent)
			reviews=self.reviewSubset
		elif whatToUse=='percentile':
			self.getWordLengthPercentile(percentile, percent)
			reviews=self.reviewSubset
		else:
			print('unsupported keyword: whatToUse')
			return

		print(len(reviews),' reviews to analyze')
		# we're interested in the text of each review 
		# and the stars rating, so we load these into 
		# separate lists
		texts = [review['text'] for review in reviews]
		stars = [review['stars'] for review in reviews]

		Testtexts = [review['text'] for review in self.testReviews]
		Teststars = [review['stars'] for review in self.testReviews]

		print(Counter(stars))
		balanced_x, balanced_y = self.balance_classes(texts, stars)
		print(Counter(balanced_y))

		all_x=balanced_x+Testtexts
		unbalanced_x,unbalanced_y = texts, stars
		# print(Counter(balanced_y))

		# This vectorizer breaks text into single words and bi-grams
		# and then calculates the TF-IDF representation
		vectorizer = TfidfVectorizer(ngram_range=(1,2))
		# vectorizer2 = TfidfVectorizer(ngram_range=(1,2),min_df=5)		#needs at least 2 occurances
		t1 = datetime.datetime.now(tz=None)

		# countVectorizer = CountVectorizer(ngram_range=(1,2))
		
		# print(vectorizer)

		# the 'fit' builds up the vocabulary from all the reviews
		# while the 'transform' step turns each indivdual text into
		# a matrix of numbers.
		vectors = vectorizer.fit_transform(balanced_x)
		# tfidf_train_data = vec.fit_transform(dataTrain) 
		tfidf_test_data = vectorizer.transform(Testtexts)
		print('Number of distinct bigrams: ',len(vectorizer.get_feature_names()))
		# print('Number of distinct bigrams: ',len(vectorizer.get_feature_names()))
		# vectors2 = vectorizer2.fit_transform(balanced_x)

		#  print(vectors)
		if showDetails:
			print('form to classifier time: ',datetime.datetime.now(tz=None) - t1)

	

		X_train, X_test, y_train, y_test = train_test_split(vectors, balanced_y, test_size=0, random_state=42)
		# X_train0, X_test, y_train0, y_test = train_test_split(vectors, Teststars, test_size=1, random_state=42)
		
		# X_test, y_test = train_test_split(vectors, testReviews, test_size=5000, random_state=42)
		
		# X_train2, X_test2, y_train2, y_test2 = train_test_split(vectors2, balanced_y, test_size=0.33, random_state=42)



		# initialise the SVM classifier
		classifier = LinearSVC()
		# classifier2= LinearSVC()
		# train the classifier
		t1 = datetime.datetime.now()
		classifier.fit(X_train, y_train)
		# classifier2.fit(X_train2, y_train2)
		if showDetails:
			print('Train time: ',datetime.datetime.now() - t1)

		preds = classifier.predict(tfidf_test_data)
		# preds2 = classifier2.predict(X_test2)
		if showDetails:
			print(list(preds[:10]))
			print(Teststars[:10])

		# print(texts[0])
		# print(texts[4])

		print('MinFreq=1 Accuracy with ', percent, ' of the data is: ',accuracy_score(Teststars, preds))
		# print('MinFreq=5 Accuracy with ', percent, ' of the data is: ',accuracy_score(y_test2, preds2))
		if showDetails:
			print(classification_report(Teststars, preds))
			print(confusion_matrix(Teststars, preds))
		return accuracy_score(Teststars, preds)


# YSVM=YelpSVM()
# YSVM.getData()	#load all data set
# YSVM.testData(5000)	#Build Test data set and remove from all data set
# # YSVM.getMostWords(.1)
# # YSVM.getLeastWords(.1)
# # # YSVM.run(.25, False)
# # # YSVM.run(.5, False)
# # # YSVM.run(.25, False)
# # # YSVM.run(.125, False)
# # # YSVM.run(.0625, False)
# YSVM.run(.1, False,'percentile', .5)
# YSVM.testData(5000)
# # YSVM.getMostWords(.1)
# YSVM.run(.1, False,'percentile', .1)
# YSVM.testData(5000)
# YSVM.run(.1, False,'percentile', .9)

# YSVM.run(.1, False,'longestReviews')
# YSVM.run(.1, False,'shortestReviews')



# print(len(getXPercent(.01)))

