import os,sys
import os.path
import pandas as pd
import numpy as np
sys.path.append(os.path.abspath(os.pardir))

from configs.settings import *
from data_access.mongo_utils import MongoDBUtils
import time
from tabulate import tabulate
from data_access.mongo_utils import MongoDBUtils

from sklearn.ensemble import RandomForestClassifier, VotingClassifier,BaggingClassifier,AdaBoostClassifier
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
import numpy as np
from sklearn.preprocessing import LabelEncoder
import ml_utils as ml_utils
from nlp_features import customStopwords

def fit_multiple_estimators(classifiers, X_list,y, sample_weights = None):

    # Convert the labels `y` using LabelEncoder, because the predict method is using index-based pointers

    # Fit all estimators with their respective feature arrays
    estimators = [clf.fit(X, y) for clf, X in zip(classifiers, X_list)]

    return estimators

def predictClass(clfs,X_list):
	classes =  np.asarray([clf.predict(X) for clf, X in zip(clfs, X_list)])
	return vote(classes)

def most_common(lst):
    return max(set(lst), key=lst.count)

def vote(allVotes):
	size = allVotes.shape[1]
	print size
	ret = []
	print allVotes	
	for index in range(size):
		tmp_list = [allVotes[0][index],allVotes[1][index],allVotes[2][index],allVotes[3][index]]
		ret.append(most_common(tmp_list))
	print ret	
	return ret	

def ensembleEverything(typeOp,balanced):
	if balanced == 'balanced':
		train_data=pd.read_csv(DATASET_PATH+"/tweets_balanced_train.csv", sep=",",dtype=str)
		test_data=pd.read_csv(DATASET_PATH+"/tweets_balanced_test.csv", sep=",",dtype=str)
	else:
		train_data=pd.read_csv(DATASET_PATH+"/"+typeOp+"_tweets_train.csv", sep=",",dtype=str)
		test_data=pd.read_csv(DATASET_PATH+"/"+typeOp+"_tweets_test.csv", sep=",",dtype=str)
	final_train_data = train_data
	final_test_data = test_data
	print final_train_data
	final_test_data = final_test_data.fillna('')
	frames = [final_train_data, final_test_data]
	df_complete= pd.concat(frames)
	train_data_sgd=final_train_data[['screen_name','tweets','age']]
	test_data_sgd=final_test_data[['screen_name','tweets','age']]

	print 'Number of observations in the training data:', len(final_train_data)
	print 'Number of observations in the test data:',len(final_test_data)
	frames = [train_data_sgd, test_data_sgd]
	df_complete= pd.concat(frames)
	print 'Number of observations in the whole dataset:',len(df_complete)
	stopwords = getCustomStopwords()  
	transformer_tfidf = TfidfVectorizer(smooth_idf=False,lowercase=False,stop_words=stopwords,max_features=5000, ngram_range=(1,3))
	tfidf = transformer_tfidf.fit_transform(final_train_data.tweets)
	train_data_features_ngrams = tfidf.toarray()
	#print ml_utils.SVM_param_selection(train_data_features_sub, final_train_data["age"]) {'kernel': 'rbf', 'C': 8, 'gamma': 1}
	#ensemble
	print "train ensemble"
	eclf1 = clf = AdaBoostClassifier(n_estimators=500,
                         learning_rate=1,
                         random_state=0)
	eclf1 = eclf1.fit(train_data_features_ngrams, final_train_data['age'])
	print "finish training ensemble"
	#test

	#sgd
	transformer_tfidf = TfidfVectorizer(smooth_idf=False,lowercase=False,stop_words=stopwords,max_features=5000, ngram_range=(1,3))
	tfidf = transformer_tfidf.fit_transform(test_data_sgd.tweets)
	test_data_features_ngrams = tfidf.toarray()
	#svm
	
	#ensamble
	featues_test = [test_data_features_ngrams,test_data_features_ngrams, test_data_features_ngrams,test_data_features_ngrams]
	y_pred = eclf1.predict(test_data_features_ngrams)
	from sklearn.metrics import accuracy_score
	y_complete = ml_utils.convertToInt(df_complete['age'],typeOp)
	db_access = MongoDBUtils()
	ageRanges = []
   	if typeOp=='normal':
   		ageRanges=db_access.getAgeRanges()
   	else:
   		ageRanges=db_access.get3AgeRanges()

	outdir =time.strftime("%d-%m-%Y")
	
	if not os.path.exists(outdir):
   		os.mkdir(outdir)

   	if not os.path.exists(outdir +"/"+typeOp):
   		os.mkdir(outdir +"/"+typeOp)

   	outdir=outdir +"/"+typeOp
   	name_prefix = "Ensemble_"
	target_names = ageRanges	   
	ml_utils.createConfusionMatrix(final_test_data['age'].tolist(),y_pred,ageRanges,name_prefix,'Ensemble',outdir)
	print classification_report(final_test_data['age'].tolist(), y_pred, target_names=target_names)

	print(accuracy_score(test_data_sgd["age"].tolist(), y_pred))


ensembleEverything('normal', 'unbalanced')