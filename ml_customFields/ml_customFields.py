
import os,sys
import os.path
import numpy as np
import pandas as pd
sys.path.append(os.path.abspath(os.pardir))
from sklearn.model_selection import train_test_split
from configs.settings import *
from data_access.mongo_utils import MongoDBUtils
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import MultinomialNB
import itertools
import time
import matplotlib.pyplot as plt

def main_customFields():
	## *********ARMO EL DATASET DE TRAIN Y EL DE TEST *********
	db_access = MongoDBUtils()
	users_df = db_access.get_customFields()

	# Split into training and test set
	# 80% of the input for training and 20% for testing

	train_data=users_df.sample(frac=0.8,random_state=200) 
	test_data=users_df.drop(train_data.index)

	# Show the number of observations for the test and training dataframes
	print 'Number of observations in the training data:', len(train_data)
	print 'Number of observations in the test data:',len(test_data)
		
	# Create a list of the feature column's names (everything but the screen_name and age)
	features = users_df.columns[1:(len(users_df.columns)-1)]

	import ml_utils as ml_utils

	# convert age ranges into integers
	y = ml_utils.convertToInt(train_data['age'])

	# Create a random forest Classifier.
	rforest = RandomForestClassifier(n_jobs=2, random_state=0)
	
	# Initialize Multinomial Naive Bayes
	bayes = MultinomialNB()

	# Train the Classifier to take the training features and learn how they relate to the age
	rforest.fit(train_data[features], y)
	bayes.fit( train_data[features], y ) 

	# Apply the Classifier we trained to the test data
	# Create actual english names for the ages for each predicted age range
	resultForest = ml_utils.convertToCategory(rforest.predict(test_data[features]))
	resultBayes = ml_utils.convertToCategory(bayes.predict(test_data[features]))
	
	# View the predicted probabilities of the first 10 observations
	rforest.predict_proba(test_data[features])[0:10]

	#############################################
	# EVALUATE THE MODEL
	#############################################

	outdir =time.strftime("%d-%m-%Y")
	
	if not os.path.exists(outdir):
   		os.mkdir(outdir)

   	##RFOREST
   	#########

	#create confusion matrix: anything on the diagonal was classified correctly and the rest incorrectly.
	cnf_matrix =confusion_matrix(test_data['age'].tolist(), resultForest)
	#print "Confusion Matrix: ", cnf_matrix

	# Plot non-normalized confusion matrix
	fig2 = plt.figure()
	ml_utils.plot_confusion_matrix(cnf_matrix, classes=db_access.getAgeRanges(),
	                    title='Confusion matrix, without normalization for custom fields')
	
	outname = 'ml_customFields_RF_confusionMatrixNotNormalized.png'
	fullname = os.path.join(outdir, outname)    
	fig2.savefig(fullname)

	# Plot normalized confusion matrix
	fig3 = plt.figure()
	ml_utils.plot_confusion_matrix(cnf_matrix, classes=db_access.getAgeRanges(), normalize=True,
                    title='Normalized confusion matrix for custom fields')
	
	outname = 'ml_customFields_RF_confusionMatrixNormalized.png'
	fullname = os.path.join(outdir, outname)
	fig3.savefig(fullname)

	##BAYES
	########

	#create confusion matrix: anything on the diagonal was classified correctly and the rest incorrectly.
	cnf_matrix2 =confusion_matrix(test_data['age'].tolist(), resultBayes)
	#print "Confusion Matrix: ", cnf_matrix2

	# Plot non-normalized confusion matrix
	fig4 = plt.figure()
	ml_utils.plot_confusion_matrix(cnf_matrix2, classes=db_access.getAgeRanges(),
	                    title='Confusion matrix, without normalization for custom fields')
	
	outname = 'ml_customFields_BAYES_confusionMatrixNotNormalized.png'
	fullname = os.path.join(outdir, outname)    
	fig4.savefig(fullname)

	# Plot normalized confusion matrix
	fig5 = plt.figure()
	ml_utils.plot_confusion_matrix(cnf_matrix2, classes=db_access.getAgeRanges(), normalize=True,
                    title='Normalized confusion matrix for custom fields')
	
	outname = 'ml_customFields_BAYES_confusionMatrixNormalized.png'
	fullname = os.path.join(outdir, outname)
	fig5.savefig(fullname)

	# View a list of the features and their importance scores
	print "Importance of Features: ", list(zip(train_data[features], rforest.feature_importances_))

	# Copy the results to a pandas dataframe 
	output = pd.DataFrame( data={"id":test_data["screen_name"], "realAge":test_data["age"], "ageRandomForest":resultForest,"ageNaiveBayes":resultBayes})
	#print output

	outname = 'ml_customFields_result.csv'
	fullname = os.path.join(outdir, outname)    
	output.to_csv(fullname,index=False)

	accuracyRF = accuracy_score(test_data['age'].tolist(), resultForest)
	accuracyNB = accuracy_score(test_data['age'].tolist(), resultBayes)

	print "Bayes:",accuracyNB,"|RForest:", accuracyRF
	return "Bayes:",accuracyNB,"|RForest:", accuracyRF

if __name__ == '__main__':
    main_customFields()

    #https://chrisalbon.com/machine-learning/random_forest_classifier_example_scikit.html
    #http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html#sphx-glr-auto-examples-model-selection-plot-confusion-matrix-py
