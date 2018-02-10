# coding=utf-8
# This Python file uses the following encoding: utf-8
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
from sklearn.neural_network import MLPClassifier
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
from sklearn.feature_extraction.text import TfidfVectorizer

def main_featBOW(typeOp):
	
	train_data=pd.read_csv(DATASET_PATH+"/"+typeOp+"_tweets_train.csv", sep=",",dtype=str)
	test_data=pd.read_csv(DATASET_PATH+"/"+typeOp+"_tweets_test.csv", sep=",",dtype=str)

	##STOPWORDS EN SPANISH, SCIKIT TRAE SOLO EN INGLES
	foo = imp.load_source('stopwords', DIR_PREFIX+'/proyectos/TesisVT/nlp_features/nlp_utils.py')
	stopwords = foo.generateCustomStopwords()  

	#count_vect = CountVectorizer(stop_words=stopwords, max_features=5000 ) #Para hacer bag of words
	#X_train_counts = count_vect.fit_transform(train_data.tweets)
	# fit_transform() fits the model and learns the vocabulary; second, it transforms our training data
	# into feature vectors. 

	transformer_tfidf = TfidfVectorizer(smooth_idf=False,lowercase=False,stop_words=stopwords,max_features=5000)
	tfidf = transformer_tfidf.fit_transform(train_data.tweets)

	indices = np.argsort(transformer_tfidf.idf_)[::-1]
	features = transformer_tfidf.get_feature_names()
	top_n = 500
	idf = transformer_tfidf.idf_

	valuesTfIdf = sorted(zip(idf,transformer_tfidf.get_feature_names()), key=lambda x: x[0])
	print valuesTfIdf
	#print(tabulate(values, headers, tablefmt="plain"))

	## ********* APLICO BAG OF WORDS *********
	##To see occurrences of a specific word:
	#print count_vect.vocabulary_.get(u'amigos')

	train_data_features = tfidf.toarray()
	#print len(train_data) #186 users en train

	#print train_data_features.shape 
	#(186, 500) --> It has 212 rows and 500 features (500 most frequent words).

	# Take a look at the words in the vocabulary
	vocab = transformer_tfidf.get_feature_names()
	#print vocab

	# Sum up the counts of each vocabulary word
	dist = np.sum(train_data_features, axis=0)

	# For each, print the vocabulary word and the number of times it 
	# appears in the training set
	#for tag, count in zip(vocab, dist):
	#	print count, tag

	# ********* APLICO RANDOM FOREST Y LO ENTRENO CON LA DATA EN TRAIN*********#

	print "Training the random forest..."

	# Initialize Multinomial Naive Bayes
	bayes = MultinomialNB()

	# Initialize a Random Forest classifier with 100 trees
	forest = RandomForestClassifier(n_estimators = 100) 
	# Fit the forest to the training set, using the bag of)

	svm = LinearSVC(loss='hinge', penalty='l2', random_state=42)
	
	sgd = SGDClassifier(loss='hinge', penalty='l2', random_state=42, alpha=0.001)

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

	# Use the random forest to make age range predictions
	resultForest = forest.predict(test_data_features)

	resultBayes = bayes.predict(test_data_features)

	resultSVM= svm.predict(test_data_features)

	resultSGD= sgd.predict(test_data_features)

	#clfMLP = MLPClassifier(hidden_layer_sizes=(100,100,100), max_iter=10000, alpha=0.0001,
    #                 solver='sgd', verbose=10,  random_state=21,tol=0.000000001)

	#clfMLP.fit(train_data_features, train_data["age"])
	#resultMLP = clfMLP.predict(test_data_features)

	# Copy the results to a pandas dataframe with an "id" column and
	# a "age" column

	outdir =time.strftime("%d-%m-%Y")
	
	if not os.path.exists(outdir):
   		os.mkdir(outdir)

   	if not os.path.exists(outdir +"/"+typeOp):
   		os.mkdir(outdir +"/"+typeOp)

   	outdir=outdir +"/"+typeOp

	output = pd.DataFrame( data={"id":test_data["screen_name"], "realAge":test_data["age"], "ageRandomForest":resultForest,"ageNaiveBayes":resultBayes})
	#print output

	# Use pandas to write the comma-separated output file
	outname = 'Bag_of_Words_model_ForestAndBayes.csv'
	fullname = os.path.join(outdir, outname)    
	output.to_csv(fullname,index=False)

	outname = 'Bag_of_Words_Tweets_TfIdf.csv'
	fullname = os.path.join(outdir, outname)  

	df_tfidf= pd.DataFrame(valuesTfIdf, columns = ["score", "word"]) 
	df_tfidf.to_csv(fullname,index=False)

	# View a list of the features and their importance scores
	#print "Importance of Features: ", sort(zip(train_data_features, forest.feature_importances_))

	#headers = ["name", "score"]
	#values = sorted(zip(vocab, forest.feature_importances_), key=lambda x: x[1] * -1)
	#print(tabulate(values, headers, tablefmt="plain"))

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
	ml_utils.createConfusionMatrix(test_data['age'].tolist(),resultForest,ageRanges,'featBOW','RandomForest',outdir)
	
	##BAYES
	ml_utils.createConfusionMatrix(test_data['age'].tolist(),resultBayes,ageRanges,'featBOW','NaiveBayes',outdir)
	
	##SVM
	ml_utils.createConfusionMatrix(test_data['age'].tolist(),resultSVM,ageRanges,'featBOW','SVM',outdir)
	
	##SGD
	ml_utils.createConfusionMatrix(test_data['age'].tolist(),resultSGD,ageRanges,'featBOW','SGD',outdir)
	
	##NEURAL NETWORK
	#ml_utils.createConfusionMatrix(test_data['age'].tolist(),resultMLP,ageRanges,'featBOW','NeuralNetwork')
	

	accuracyRF = accuracy_score(test_data['age'].tolist(), resultForest)
	accuracyNB = accuracy_score(test_data['age'].tolist(), resultBayes)
	#accuracyMLP = accuracy_score(test_data['age'].tolist(), predsMLP)
	accuracySVM = accuracy_score(test_data['age'].tolist(), resultSVM)
	accuracySGD = accuracy_score(test_data['age'].tolist(), resultSGD)

	fscoreRF = f1_score(test_data['age'].tolist(), resultForest, average=None, labels=ageRanges)
	fscoreNB = f1_score(test_data['age'].tolist(), resultBayes, average=None, labels=ageRanges)
	#fscoreMLP = f1_score(test_data['age'].tolist(), predsMLP, average=None, labels=ageRanges)
	fscoreSVM = f1_score(test_data['age'].tolist(), resultSVM, average=None, labels=ageRanges)
	fscoreSGD = f1_score(test_data['age'].tolist(), resultSGD, average=None, labels=ageRanges)

	print "ACCURACY--> Bayes:",accuracyNB,"|RForest:", accuracyRF,"|SVM:", accuracySVM,"|SGD:", accuracySGD#,"|NeuralN:", accuracyMLP
	print "F-SCORE--> Bayes:",fscoreNB,"|RForest:", fscoreRF,"|SVM:", fscoreSVM,"|SGD:", fscoreSGD#,"|NeuralN:", fscoreMLP
	
	return "ACCURACY--> Bayes:",accuracyNB,"|RForest:", accuracyRF,"|SVM:", accuracySVM,"|SGD:", accuracySGD,"F-SCORE--> Bayes:",fscoreNB,"|RForest:", fscoreRF,"|SVM:", fscoreSVM,"|SGD:", fscoreSGD#,"|NeuralN:", fscoreMLP#,"|NeuralN:", accuracyMLP

if __name__ == '__main__':
    main_featBOW()


