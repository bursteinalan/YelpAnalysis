import urllib.request
import json
import dml
import prov.model
import datetime
import uuid
from collections import Counter
# read the data from disk and split into lines
# we use .strip() to remove the final (empty) line

def getData():
	client = dml.pymongo.MongoClient()
	repo = client.repo
	repo.authenticate('alanbur', 'alanbur')
	collection=repo['yelp.reviews'].find().limit(100);
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