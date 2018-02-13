
import os,sys
import os.path
import numpy as np
import pandas as pd
sys.path.append(os.path.abspath(os.pardir))
from sklearn.model_selection import train_test_split
from configs.settings import *
from data_access.mongo_utils import MongoDBUtils
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm, datasets
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.naive_bayes import MultinomialNB
from stop_words import get_stop_words
import re
import imp
import time
from nltk.corpus import stopwords
from tabulate import tabulate
import matplotlib.pyplot as plt
from sklearn.svm import LinearSVC
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import f1_score
from nlp_features.customStopwords import getSpanishStopwords

def main_subscriptionBOW(typeOp,balanced):
	
	if balanced == 'balanced':
		train_data=pd.read_csv(DATASET_PATH+"/subscriptionLists_balanced_train.csv", sep=",",dtype=str)
		test_data=pd.read_csv(DATASET_PATH+"/subscriptionLists_balanced_train.csv", sep=",",dtype=str)
	else:
		train_data=pd.read_csv(DATASET_PATH+"/"+typeOp+"_subscriptionLists_train.csv", sep=",",dtype=str)
		test_data=pd.read_csv(DATASET_PATH+"/"+typeOp+"_subscriptionLists_test.csv", sep=",",dtype=str)

	print train_data.shape
	print test_data.shape

	stopwords = getSpanishStopwords()  

	count_vect = CountVectorizer(stop_words=stopwords, max_features=5000,ngram_range=(1,3), token_pattern=r'\b\w+\b' ) #Para hacer bag of words
	X_train_counts = count_vect.fit_transform(train_data.subscriptionLists)
	# fit_transform() fits the model and learns the vocabulary; second, it transforms our training data
	# into feature vectors. 

	## ********* APLICO BAG OF WORDS *********
	##To see occurrences of a specific word:
	#print count_vect.vocabulary_.get(u'amigos')

	train_data_features = X_train_counts.toarray()
	#print len(train_data) #186 users en train

	#print train_data_features.shape 
	#(186, 500) --> It has 212 rows and 500 features (500 most frequent words).

	# Take a look at the words in the vocabulary
	vocab = count_vect.get_feature_names()
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
	#print ml_utils.SVM_param_selection(train_data_features, train_data["age"]) #RESULT:{'kernel': 'rbf', 'C': 10, 'gamma': 0.01}
	#print ml_utils.RandomForest_param_selection(train_data_features, train_data["age"])#RESULT: {'n_estimators': 120, 'max_depth': 30, 'min_samples_leaf': 1}
	#print ml_utils.SGD_param_selection(train_data_features, train_data["age"]) #RESULT: {'penalty': 'elasticnet', 'alpha': 0.0001, 'n_iter': 40, 'loss': 'log'}
	

	# ********* APLICO MODELOS Y LOS ENTRENO CON LA DATA EN TRAIN*********#

	print "Training the models..."

	# Initialize Multinomial Naive Bayes
	bayes = MultinomialNB()

	# Initialize a Random Forest classifier with 100 trees
	forest = RandomForestClassifier(n_estimators=120, max_depth= 30, min_samples_leaf= 1) 
	# Fit the forest to the training set, using the bag of)

	svm = SVC(kernel='rbf', C= 10, gamma= 0.01)
	
	sgd = SGDClassifier(penalty = elasticnet, alpha=0.0001, n_iter=40, loss='log')

	# Fit the forest to the training set, using the bag of words as 
	# features and the age range as the response variable

	forest = forest.fit( train_data_features, train_data["age"] ) 

	bayes = bayes.fit( train_data_features, train_data["age"] ) 

	svm = svm.fit(train_data_features, train_data["age"] ) 

	sgd= sgd.fit(train_data_features, train_data["age"] ) 
	# Read the test data

	# Get a bag of words for the test set, and convert to a numpy array
	test_data_features = count_vect.transform(test_data.subscriptionLists)
	test_data_features = test_data_features.toarray()

	# Use the random forest to make age range predictions
	resultForest = forest.predict(test_data_features)

	resultBayes = bayes.predict(test_data_features)

	resultSVM= svm.predict(test_data_features)

	resultSGD= sgd.predict(test_data_features)

	outdir =time.strftime("%d-%m-%Y")
	
	if not os.path.exists(outdir):
   		os.mkdir(outdir)

   	if not os.path.exists(outdir +"/"+typeOp):
   		os.mkdir(outdir +"/"+typeOp)

   	outdir=outdir +"/"+typeOp

	output = pd.DataFrame( data={"id":test_data["screen_name"], "realAge":test_data["age"], "ageRandomForest":resultForest,"ageNaiveBayes":resultBayes})
	#print output

	# Use pandas to write the comma-separated output file
	outname = 'subscriptionLists_Bag_of_Words_ForestAndBayes.csv'
	fullname = os.path.join(outdir, outname)    
	output.to_csv(fullname,index=False)

	# View a list of the features and their importance scores
	#print "Importance of Features: ", sort(zip(train_data_features, forest.feature_importances_))

	headers = ["name", "score"]
	values = sorted(zip(vocab, forest.feature_importances_), key=lambda x: x[1] * -1)
	print(tabulate(values[:100], headers, tablefmt="plain"))

	###################################
	#******* MODEL EVALUATION *********
	###################################
	import ml_utils as ml_utils
	db_access = MongoDBUtils()
	
	ageRanges=[]
   	if typeOp=='normal':
   		ageRanges=db_access.getAgeRanges()
   	else:
   		ageRanges=db_access.get3AgeRanges()

	##RANDOM FOREST
	ml_utils.createConfusionMatrix(test_data['age'].tolist(),resultForest,ageRanges,'subscriptionBOW','RandomForest',outdir)
	
	##BAYES
	ml_utils.createConfusionMatrix(test_data['age'].tolist(),resultBayes,ageRanges,'subscriptionBOW','NaiveBayes',outdir)
	
	##SVM
	ml_utils.createConfusionMatrix(test_data['age'].tolist(),resultSVM,ageRanges,'subscriptionBOW','SVM',outdir)
	
	##SGD
	ml_utils.createConfusionMatrix(test_data['age'].tolist(),resultSGD,ageRanges,'subscriptionBOW','SGD',outdir)

	accuracyRF = accuracy_score(test_data['age'].tolist(), resultForest)
	accuracyNB = accuracy_score(test_data['age'].tolist(), resultBayes)
	accuracySVM = accuracy_score(test_data['age'].tolist(), resultSVM)
	accuracySGD = accuracy_score(test_data['age'].tolist(), resultSGD)

	fscoreRF = f1_score(test_data['age'].tolist(), resultForest, average=None, labels=ageRanges)
	fscoreNB = f1_score(test_data['age'].tolist(), resultBayes, average=None, labels=ageRanges)
	fscoreSVM = f1_score(test_data['age'].tolist(), resultSVM, average=None, labels=ageRanges)
	fscoreSGD = f1_score(test_data['age'].tolist(), resultSGD, average=None, labels=ageRanges)

	print "ACCURACY--> Bayes:",accuracyNB,"|RForest:", accuracyRF,"|SVM:", accuracySVM,"|SGD:", accuracySGD
	print "F-SCORE--> Bayes:",fscoreNB,"|RForest:", fscoreRF,"|SVM:", fscoreSVM,"|SGD:", fscoreSGD
	
	return "ACCURACY--> Bayes:",accuracyNB,"|RForest:", accuracyRF,"|SVM:", accuracySVM,"|SGD:", accuracySGD,"F-SCORE--> Bayes:",fscoreNB,"|RForest:", fscoreRF,"|SVM:", fscoreSVM,"|SGD:", fscoreSGD

if __name__ == '__main__':
    main_subscriptionBOW()

