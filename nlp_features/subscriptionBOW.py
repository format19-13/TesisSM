
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
from sklearn.neural_network import MLPClassifier
import re
import imp
import time
from nltk.corpus import stopwords
from tabulate import tabulate
import matplotlib.pyplot as plt

def main_subscriptionBOW():

	## *********ARMO EL DATASET DE TRAIN Y EL DE TEST *********
	db_access = MongoDBUtils()
	users_df = db_access.get_SubscriptionLists()
	print (users_df.size)
	# Split into training and test set
	# 80% of the input for training and 20% for testing

	train_data=users_df.sample(frac=0.8,random_state=200) 
	test_data=users_df.drop(train_data.index)

	##STOPWORDS EN SPANISH, SCIKIT TRAE SOLO EN INGLES
	foo = imp.load_source('stopwords', DIR_PREFIX+'/proyectos/TesisVT/nlp_features/nlp_utils.py')
	stopwords = foo.generateCustomStopwords()  

	count_vect = CountVectorizer(stop_words=stopwords, max_features=500 ) #Para hacer bag of words
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

	# ********* APLICO RANDOM FOREST Y LO ENTRENO CON LA DATA EN TRAIN*********#

	print "Training the random forest..."

	# Initialize Multinomial Naive Bayes
	bayes = MultinomialNB()

	# Initialize a Random Forest classifier with 100 trees
	forest = RandomForestClassifier(n_estimators = 100) 
	# Fit the forest to the training set, using the bag of)

	# Fit the forest to the training set, using the bag of words as 
	# features and the age range as the response variable

	forest = forest.fit( train_data_features, train_data["age"] ) 

	bayes = bayes.fit( train_data_features, train_data["age"] ) 

	# Read the test data

	# Get a bag of words for the test set, and convert to a numpy array
	test_data_features = count_vect.transform(test_data.subscriptionLists)
	test_data_features = test_data_features.toarray()

	# Use the random forest to make age range predictions
	resultForest = forest.predict(test_data_features)

	resultBayes = bayes.predict(test_data_features)

	clfMLP = MLPClassifier(hidden_layer_sizes=(100,100,100), max_iter=10000, alpha=0.0001,
                     solver='sgd', verbose=10,  random_state=21,tol=0.000000001)

	clfMLP.fit(train_data_features, train_data["age"])
	predsMLP = clfMLP.predict(test_data_features)

	# Copy the results to a pandas dataframe with an "id" column and
	# a "age" column

	outdir =time.strftime("%d-%m-%Y")
	
	if not os.path.exists(outdir):
   		os.mkdir(outdir)

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
	print(tabulate(values, headers, tablefmt="plain"))

	###################################
	#******* MODEL EVALUATION *********
	###################################
	import ml_utils as ml_utils

	##RFOREST
	###############

	#create confusion matrix: anything on the diagonal was classified correctly and the rest incorrectly.
	cnf_matrix =confusion_matrix(test_data['age'].tolist(), resultForest)
	#print "Confusion Matrix for Random Forest: "
	#print cnf_matrix

	# Plot non-normalized confusion matrix
	fig2 = plt.figure()
	ml_utils.plot_confusion_matrix(cnf_matrix, classes=db_access.getAgeRanges(),
	                    title='Confusion matrix, without normalization for Subscription BOW - Random Forest')
	
	outname = 'ml_subscriptionBOW_randomForest_confusionMatrixNotNormalized.png'
	fullname = os.path.join(outdir, outname)    
	fig2.savefig(fullname)

	# Plot normalized confusion matrix
	fig3 = plt.figure()
	ml_utils.plot_confusion_matrix(cnf_matrix, classes=db_access.getAgeRanges(), normalize=True,
                    title='Normalized confusion matrix for Subscription BOW - Random Forest')
	
	outname = 'ml_subscriptionBOW_randomForest_confusionMatrixNormalized.png'
	fullname = os.path.join(outdir, outname)
	fig3.savefig(fullname)

	##BAYES
	###############

	cnf_matrix2 =confusion_matrix(test_data['age'].tolist(), resultBayes)
	#print "Confusion Matrix for Naive Bayes: "
	#print cnf_matrix2

	# Plot non-normalized confusion matrix
	fig2 = plt.figure()
	ml_utils.plot_confusion_matrix(cnf_matrix2, classes=db_access.getAgeRanges(),
	                    title='Confusion matrix, without normalization for Subscription BOW - Bayes')
	
	outname = 'ml_subscriptionBOW_Bayes_confusionMatrixNotNormalized.png'
	fullname = os.path.join(outdir, outname)    
	fig2.savefig(fullname)

	# Plot normalized confusion matrix
	fig3 = plt.figure()
	ml_utils.plot_confusion_matrix(cnf_matrix2, classes=db_access.getAgeRanges(), normalize=True,
                    title='Normalized confusion matrix for Subscription BOW - Bayes')
	
	outname = 'ml_subscriptionBOW_Bayes_confusionMatrixNormalized.png'
	fullname = os.path.join(outdir, outname)
	fig3.savefig(fullname)

	##NEURAL NETWORK
	###############

	cnf_matrix3 =confusion_matrix(test_data['age'].tolist(), predsMLP)
	#print "Confusion Matrix for Neural Network: "
	#print cnf_matrix3

	# Plot non-normalized confusion matrix
	fig4 = plt.figure()
	ml_utils.plot_confusion_matrix(cnf_matrix3, classes=db_access.getAgeRanges(),
	                    title='Confusion matrix, without normalization for Feat BOW - NeuralNetwork')
	
	outname = 'ml_featBOW_NeuralN_confusionMatrixNotNormalized.png'
	fullname = os.path.join(outdir, outname)    
	fig4.savefig(fullname)

	# Plot normalized confusion matrix
	fig5 = plt.figure()
	ml_utils.plot_confusion_matrix(cnf_matrix3, classes=db_access.getAgeRanges(), normalize=True,
                    title='Normalized confusion matrix for Feat BOW - NeuralNetwork')
	
	outname = 'ml_featBOW_NeuralN_confusionMatrixNormalized.png'
	fullname = os.path.join(outdir, outname)
	fig5.savefig(fullname)

	accuracyRF = accuracy_score(test_data['age'].tolist(), resultForest)
	accuracyNB = accuracy_score(test_data['age'].tolist(), resultBayes)
	accuracyMLP = accuracy_score(test_data['age'].tolist(), predsMLP)

	return "Bayes:",accuracyNB,"|RForest:", accuracyRF,"|NeuralN:", accuracyMLP

if __name__ == '__main__':
    main_subscriptionBOW()

