from sklearn.datasets.samples_generator import make_blobs
from sklearn.datasets.samples_generator import make_circles
from sklearn.datasets.samples_generator import make_moons
from sklearn.datasets.samples_generator import make_s_curve
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from pprint import pprint

import collections

# coding=utf-8
# This Python file uses the following encoding: utf-8
import os,sys
import os.path
import pandas as pd
import numpy as np
sys.path.append(os.path.abspath(os.pardir))

from configs.settings import *
from data_access.mongo_utils import MongoDBUtils
import time
from tabulate import tabulate

from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier, LinearRegression
from sklearn.svm import SVC

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from nlp_features.customStopwords import getCustomStopwords

from sklearn.metrics import accuracy_score,make_scorer,classification_report
from sklearn.model_selection import cross_val_score,StratifiedKFold
from sklearn import metrics
from math import sqrt

def main_clustering(typeOp,balanced):
	
	if balanced == 'balanced':
		train_data=pd.read_csv(DATASET_PATH+"/tweets_balanced_train.csv", sep=",",dtype=str)[['screen_name','tweets','age']]
		test_data=pd.read_csv(DATASET_PATH+"/tweets_balanced_test.csv", sep=",",dtype=str)[['screen_name','tweets','age']]
	else:
		train_data=pd.read_csv(DATASET_PATH+"/"+typeOp+"_tweets_train.csv", sep=",",dtype=str)[['screen_name','tweets','age']]
		test_data=pd.read_csv(DATASET_PATH+"/"+typeOp+"_tweets_train.csv", sep=",",dtype=str)[['screen_name','tweets','age']]

	# Show the number of observations for the test and training dataframes
	print 'Number of observations in the training data:', len(train_data)
	print 'Number of observations in the test data:',len(test_data)
	
	frames = [train_data, test_data]
	df_complete= pd.concat(frames)

	print 'Number of observations in the whole dataset:',len(df_complete)

	
	##STOPWORDS EN SPANISH, SCIKIT TRAE SOLO EN INGLES
	stopwords = getCustomStopwords()  

	#count_vect = CountVectorizer(stop_words=stopwords, max_features=5000 ) #Para hacer bag of words
	#X_train_counts = count_vect.fit_transform(train_data.tweets)
	# fit_transform() fits the model and learns the vocabulary; second, it transforms our training data
	# into feature vectors. 

	transformer_tfidf = TfidfVectorizer(smooth_idf=False,lowercase=False,stop_words=stopwords,max_features=5000, ngram_range=(1,3))
	X = transformer_tfidf.fit_transform(train_data.tweets)

    ##To see occurrences of a specific word:
	#print count_vect.vocabulary_.get(u'amigos')

	kmeans = KMeans(n_clusters = 5, random_state = 0)                   
	kmeans.fit(X)                  

	clustering = collections.defaultdict(list)
	for idx, label in enumerate(kmeans.labels_):
	    clustering[label].append(idx)
	plt.scatter(X[:,0],X[:,1], c=kmeans.labels_, cmap='rainbow')  
	plt.scatter(kmeans.cluster_centers_[:,0] ,kmeans.cluster_centers_[:,1], color='black')    
	plt.show()
	return clustering

main_clustering('normal', 'unbalanced')

