from sklearn.base import BaseEstimator
from sklearn.base import ClassifierMixin
import numpy as np
import operator
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

from sklearn.naive_bayes import MultinomialNB

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from nlp_features.customStopwords import getCustomStopwords

from sklearn.metrics import accuracy_score,make_scorer,classification_report
from sklearn.model_selection import cross_val_score,StratifiedKFold

class EnsembleClassifier(BaseEstimator, ClassifierMixin):

    def __init__(self, clf):
        self.clf = clf

    def fit(self, X, y):
        self.clf = self.clf.fit(X, y)

    def voteAge(self,x):
        ages = (x[0].split("-"))
        cotaInferior = int(ages[0])
        coteSuperior = int(ages[1])
        ageFromPic = int(x[1])
        print "datos: ", x
        print "no -1", ageFromPic != -1 
        print "cota enfrior: ", ageFromPic >= cotaInferior 
        print "cota superior: ", ageFromPic <= coteSuperior
        if ageFromPic != -1 and ageFromPic >= cotaInferior and ageFromPic <= coteSuperior:
            print "ret: ", x[0]
            return x[0]
        else:
            return "null"    

    def predict(self, X, externalVote):
        internalVote = self.clf.predict(X)
        tuples = zip(internalVote, externalVote)
        return map(self.voteAge, tuples)

def main_profilePicVote(typeOp,balanced):
	
	if balanced == 'balanced':
		train_data=pd.read_csv(DATASET_PATH+"/tweets_balanced_train.csv", sep=",",dtype=str)[['screen_name','tweets','age']]
		test_data=pd.read_csv(DATASET_PATH+"/tweets_balanced_test.csv", sep=",",dtype=str)[['screen_name','tweets','age']]
	else:
		train_data=pd.read_csv(DATASET_PATH+"/"+typeOp+"_tweets_train.csv", sep=",",dtype=str)[['screen_name','tweets','age','profile_pic_age']]
		test_data=pd.read_csv(DATASET_PATH+"/"+typeOp+"_tweets_test.csv", sep=",",dtype=str)[['screen_name','tweets','age','profile_pic_age']]



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
	tfidf = transformer_tfidf.fit_transform(train_data.tweets)


	##To see occurrences of a specific word:
	#print count_vect.vocabulary_.get(u'amigos')

	train_data_features = tfidf.toarray()
	#print len(train_data) #186 users en train

	# Take a look at the words in the vocabulary
	vocab = transformer_tfidf.get_feature_names()
	#print vocab

	# Sum up the counts of each vocabulary word
	dist = np.sum(train_data_features, axis=0)

	# For each, print the vocabulary word and the number of times it 
	# appears in the training set
	#for tag, count in zip(vocab, dist):
	#	print count, tag

	########################################
	#******* HYPERPARAMETER TUNING *********
	########################################
	
	import ml_utils as ml_utils

	print 'Training'

	# Initialize Multinomial Naive Bayes
	bayes = MultinomialNB()
	eclf = EnsembleClassifier(clf=bayes)
	eclf.fit( train_data_features, train_data["age"] ) 
	# Read the test data
	print 'Fit'

	# Get a bag of words for the test set, and convert to a numpy array
	test_data_features = transformer_tfidf.transform(test_data.tweets)
	test_data_features = test_data_features.toarray()

	# Use the random forest to make age range predictions
	#resultForest = forest.predict(test_data_features)

	resultBayes = eclf.predict(test_data_features, train_data["profile_pic_age"])
	print "resultbayesVote: ", resultBayes


main_profilePicVote('normal', 'unbalanced')