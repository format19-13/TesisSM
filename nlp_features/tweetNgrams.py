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



def main_tweetNgrams(typeOp,balanced):
	
	if balanced == 'balanced':
		train_data=pd.read_csv(DATASET_PATH+"/tweets_balanced_train.csv", sep=",",dtype=str)[['screen_name','tweets','age']]
		test_data=pd.read_csv(DATASET_PATH+"/tweets_balanced_test.csv", sep=",",dtype=str)[['screen_name','tweets','age']]
	else:
		train_data=pd.read_csv(DATASET_PATH+"/"+typeOp+"_tweets_train.csv", sep=",",dtype=str)[['screen_name','tweets','age']]
		test_data=pd.read_csv(DATASET_PATH+"/"+typeOp+"_tweets_test.csv", sep=",",dtype=str)[['screen_name','tweets','age']]

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
	#PARAMETERS TUNING
	#print ml_utils.SVM_param_selection(train_data_features, train_data["age"]) #RESULT: {'kernel': 'rbf', 'C': 10, 'gamma': 0.1} 
	#print ml_utils.RandomForest_param_selection(train_data_features, train_data["age"])#RESULT: {'n_estimators': 160, 'max_depth': 20, 'min_samples_leaf': 3}
	#print ml_utils.SGD_param_selection(train_data_features, train_data["age"]) #RESULT: {'penalty': 'elasticnet', 'alpha': 0.0001, 'n_iter': 50, 'loss': 'log'}
	
	########################################
	#******* MODEL TRAINING        *********
	########################################

	# ********* ENTRENO LOS MODELOS CON LA DATA EN TRAIN*********#

	print "Training the Classifiers..."

	# Initialize Multinomial Naive Bayes
	bayes = MultinomialNB()

	# Initialize a Random Forest classifier with 100 trees
	forest = RandomForestClassifier(n_estimators = 160, max_depth=20,min_samples_leaf=3) 
	# Fit the forest to the training set, using the bag of)

	svm = SVC(kernel='rbf', C=10, gamma= 0.1,probability=True)

	sgd = SGDClassifier(loss='log', penalty='l2', random_state=42, alpha=0.0001,n_iter=60)

	regr = LinearRegression()
	# Fit the forest to the training set, using the bag of words as 
	# features and the age range as the response variable

	forest = forest.fit( train_data_features, train_data["age"] ) 

	bayes = bayes.fit( train_data_features, train_data["age"] ) 

	svm = svm.fit(train_data_features, train_data["age"] ) 

	sgd= sgd.fit(train_data_features, train_data["age"] ) 
	
	# Read the test data

	# Get a bag of words for the test set, and convert to a numpy array
	test_data_features = transformer_tfidf.transform(test_data.tweets)
	test_data_features = test_data_features.toarray()

	#train_filtered_data = train_data[pd.to_numeric(train_data.age, errors='coerce').notnull()]
	
	#tfidf_filtered = transformer_tfidf.fit_transform(train_filtered_data.tweets)

	#train_data_features_filtered = tfidf_filtered.toarray()

	#regr = regr.fit(train_data_features_filtered, train_filtered_data["age"])

	# Use the random forest to make age range predictions
	resultForest = forest.predict(test_data_features)

	resultBayes = bayes.predict(test_data_features)
	print "resultbayes: ", resultBayes

	resultSVM= svm.predict(test_data_features)

	resultSGD= sgd.predict(test_data_features)

	#test_filtered_data = test_data[pd.to_numeric(test_data.age, errors='coerce').notnull()]
	
	#tfidf_filtered_train = transformer_tfidf.fit_transform(test_filtered_data.tweets)

	#test_data_features_filtered = tfidf_filtered_train.toarray()

	#resultLR = regr.predict(test_data_features_filtered)

	outdir =time.strftime("%d-%m-%Y")
	
	if not os.path.exists(outdir):
   		os.mkdir(outdir)

   	if not os.path.exists(outdir +"/"+typeOp):
   		os.mkdir(outdir +"/"+typeOp)

   	outdir=outdir +"/"+typeOp

	output = pd.DataFrame(data={"id":test_data["screen_name"], "realAge":test_data["age"], "ageRandomForest":resultForest,"ageNaiveBayes":resultBayes,"ageSVM":resultSVM,"ageSGD":resultSGD})

	# Use pandas to write the comma-separated output file
	outname = 'tweets_ngrams_results.csv'
	fullname = os.path.join(outdir, outname)    
	#output.to_csv(fullname,index=False)


	###################################
	#******* MODEL EVALUATION *********
	###################################

	print "Evaluating the model --> Calculating metrics ..."

	db_access = MongoDBUtils()
	
	ageRanges=[]
   	if typeOp=='normal':
   		ageRanges=db_access.getAgeRanges()
   	else:
   		ageRanges=db_access.get3AgeRanges()

	target_names = ageRanges

   	data = df_complete[['screen_name',  'tweets']]
	data = transformer_tfidf.fit_transform(data.tweets)
	y_complete = df_complete['age']


	name_prefix='tweetNgrams_'+typeOp+'_'+balanced
   	#--------------
	##BAYES
	#--------------
	print "Metrics for Naive Bayes:"
	ml_utils.createConfusionMatrix(test_data['age'].tolist(),resultBayes,ageRanges,name_prefix,'NaiveBayes',outdir)
	print classification_report(test_data['age'].tolist(), resultBayes, target_names=target_names)
	ml_utils.plotProba(bayes,"tweet_ngrams", "bayes", outdir, test_data_features)
	scores = cross_val_score(bayes, data, y_complete, cv=StratifiedKFold(n_splits=10, shuffle=True, random_state = 5),scoring=make_scorer(accuracy_score))
	accuracyNB = round(scores.mean(),2)	
	print "10-Fold Accuracy: ", accuracyNB

	#--------------
	##RANDOM FOREST
	#--------------
	print "Metrics for Random Forest:"
	ml_utils.createConfusionMatrix(test_data['age'].tolist(),resultForest,ageRanges,name_prefix,'RandomForest',outdir)
	print classification_report(test_data['age'].tolist(), resultForest, target_names=target_names)
	scores = cross_val_score(forest, data, y_complete, cv=StratifiedKFold(n_splits=10, shuffle=True, random_state = 5),scoring=make_scorer(accuracy_score))
	accuracyRF = round(scores.mean(),2)	
	ml_utils.plotProba(forest,"tweet_ngrams", "forest", outdir, test_data_features)
	print "10-Fold Accuracy: ", accuracyRF 

	#--------------
	##SVM
	#--------------
	print "Metrics for SVM:"
	ml_utils.createConfusionMatrix(test_data['age'].tolist(),resultSVM,ageRanges,name_prefix,'SVM',outdir)
	print classification_report(test_data['age'].tolist(), resultSVM, target_names=target_names)
	ml_utils.plotProba(svm,"tweet_ngrams", "svm", outdir, test_data_features)
	scores = cross_val_score(svm, data, y_complete, cv=StratifiedKFold(n_splits=10, shuffle=True, random_state = 5),scoring=make_scorer(accuracy_score))
	accuracySVM = round(scores.mean(),2)	
	print "10-Fold Accuracy: ", accuracySVM

	#--------------
	##SGD
	#--------------
	print "Metrics for SGD:"
	ml_utils.createConfusionMatrix(test_data['age'].tolist(),resultSGD,ageRanges,name_prefix,'SGD',outdir)
	print classification_report(test_data['age'].tolist(), resultSGD, target_names=target_names)
	ml_utils.plotProba(sgd,"tweet_ngrams", "sgd", outdir, test_data_features)
	scores = cross_val_score(sgd, data, y_complete, cv=StratifiedKFold(n_splits=10, shuffle=True, random_state = 5),scoring=make_scorer(accuracy_score))
	accuracySGD = round(scores.mean(),2)	
	print "10-Fold Accuracy: ", accuracySGD 

	#--------------
	##LR
	#--------------
	#print "Metrics for Lr:"]

	#print "LR, MEAN ABSOLUTE ERROR: ", metrics.mean_absolute_error(map(int,test_filtered_data['age'].tolist()),resultLR)

	#print "LR, MEAN SQUARE ERROR: ", metrics.mean_squared_error(map(int,test_filtered_data['age'].tolist()),resultLR)

	#print "LR, RMSE: ", sqrt(metrics.mean_squared_error(map(int,test_filtered_data['age'].tolist()),resultLR))


	#--------------
	##OUTPUT
	#--------------
	result= "ACCURACY--> N.Bayes:",0,"|RForest:", 0,"|SVM:", 0,"|SGD:", 0
	#print result
	return result

if __name__ == '__main__':
    main_featBOW()



