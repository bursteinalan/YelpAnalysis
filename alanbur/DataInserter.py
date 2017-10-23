#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 20 12:23:54 2017

@author: burstein
"""
import json
import dml
import prov.model
class reader(dml.Algorithm):
    
    contributor = 'alanbur'
    reads = []
    writes = ['yelp.reviews']
    @staticmethod
    def execute(trial = False):
        client = dml.pymongo.MongoClient()
        repo = client.repo
        repo.authenticate('alanbur', 'alanbur')
        
        repo.dropCollection("reviews")
        repo.createCollection("reviews")

        file='/Users/burstein/Documents/directedStudy/dataset/review.json'
        with open(file) as data_file:  
            data=[] 
            total=0
            for line in data_file:
                if len(data)==1000:
                    print('inserted: ', total)
                    repo['yelp.reviews'].insert_many(data)
                    data=[]
                # print(line)
                review = json.loads(line)
                data.append(review)
                total+=1
        
        print('inserted: ', total)
        
        repo['yelp.reviews'].insert_many(data)
        repo['yelp.reviews'].metadata({'complete':True})
        repo.logout()
        print(repo['yelp.reviews'].metadata())
    @staticmethod
    def provenance(doc = prov.model.ProvDocument(), startTime = None, endTime = None):
        client = dml.pymongo.MongoClient()
        repo = client.repo
        return doc
reader.execute()