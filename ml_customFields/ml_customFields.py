# coding=utf-8
# This Python file uses the following encoding: utf-8
import os,sys
import os.path
import pandas as pd
sys.path.append(os.path.abspath(os.pardir))

from configs.settings import *
from data_access.mongo_utils import MongoDBUtils
import time
from tabulate import tabulate

from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier
from sklearn.svm import SVC

from sklearn.metrics import accuracy_score,make_scorer,classification_report
from sklearn.model_selection import cross_val_score,StratifiedKFold

def main_customFields(typeOp,balanced):
	
	if balanced == 'balanced':
		train_data=pd.read_csv(DATASET_PATH+"/tweets_balanced_train.csv", sep=",",dtype=str)[['screen_name','friends_count',  'tweets_count', 'linkedin', 'snapchat', 'instagram','facebook','followers_count','favourites_count','qtyMentions','qtyHashtags','qtyUrls', 'qtyEmojis', 'qtyUppercase','profile_pic_gender','age']]
		test_data=pd.read_csv(DATASET_PATH+"/tweets_balanced_test.csv", sep=",",dtype=str)[['screen_name','friends_count',  'tweets_count', 'linkedin', 'snapchat', 'instagram','facebook','followers_count','favourites_count','qtyMentions','qtyHashtags','qtyUrls', 'qtyEmojis', 'qtyUppercase','profile_pic_gender','age']]
	else:
		train_data=pd.read_csv(DATASET_PATH+"/"+typeOp+"_tweets_train.csv", sep=",",dtype=str)[['screen_name','friends_count',  'tweets_count', 'linkedin', 'snapchat', 'instagram','facebook','followers_count','favourites_count','qtyMentions','qtyHashtags','qtyUrls', 'qtyEmojis', 'qtyUppercase','profile_pic_gender','age']]
		test_data=pd.read_csv(DATASET_PATH+"/"+typeOp+"_tweets_test.csv", sep=",",dtype=str)[['screen_name','friends_count',  'tweets_count', 'linkedin', 'snapchat', 'instagram','facebook','followers_count','favourites_count','qtyMentions','qtyHashtags','qtyUrls', 'qtyEmojis', 'qtyUppercase','profile_pic_gender','age']]
	
	# Show the number of observations for the test and training dataframes
	print 'Number of observations in the training data:', len(train_data)
	print 'Number of observations in the test data:',len(test_data)
		
	frames = [train_data, test_data]
	df_complete= pd.concat(frames)

	print 'Number of observations in the whole dataset:',len(df_complete)

	features = train_data.columns[1:(len(train_data.columns)-1)]
	train_data_features=train_data[features]
	test_data_features=test_data[features]

	import ml_utils as ml_utils

	# convert age ranges into integers
	y = ml_utils.convertToInt(train_data['age'],typeOp)

	########################################
	#******* HYPERPARAMETER TUNING *********
	########################################
	
	import ml_utils as ml_utils
	#PARAMETERS TUNING
	#print ml_utils.SVM_param_selection(train_data_features, train_data["age"]) #RESULT: {'kernel': 'rbf', 'C': 8, 'gamma': 0.01}
	#print ml_utils.RandomForest_param_selection(train_data_features, train_data["age"])#RESULT:{'n_estimators': 140, 'max_depth': 20, 'min_samples_leaf': 2}
	#print ml_utils.SGD_param_selection(train_data_features, train_data["age"]) #RESULT:{'penalty': 'l2', 'alpha': 0.001, 'n_iter': 50, 'loss': 'log'}

	########################################
	#******* MODEL TRAINING        *********
	########################################
	print "Training the classifiers ..."
	
	forest = RandomForestClassifier(n_estimators=140, max_depth=20,min_samples_leaf=2 )

	bayes = MultinomialNB()
	
	svm = SVC(kernel='rbf', C= 8, gamma= 0.01)
	
	sgd = SGDClassifier(loss='log', penalty='l2', random_state=42, alpha=0.001, n_iter=50) 

	# Train the Classifier to take the training features and learn how they relate to the age
	forest.fit(train_data_features, y)
	
	bayes.fit(train_data_features, y) 
	
	svm = svm.fit(train_data_features, y) 
	
	sgd= sgd.fit(train_data_features, y) 
	
	# Apply the Classifier we trained to the test data
	# Create actual english names for the ages for each predicted age range
	resultForest = ml_utils.convertToCategory(forest.predict(test_data_features),typeOp)
	
	resultBayes = ml_utils.convertToCategory(bayes.predict(test_data_features),typeOp)
	
	resultSVM= ml_utils.convertToCategory(svm.predict(test_data_features),typeOp)
	
	resultSGD= ml_utils.convertToCategory(sgd.predict(test_data_features),typeOp)
	
	# View the predicted probabilities of the first 10 observations
	forest.predict_proba(test_data_features)[0:10]

	outdir =time.strftime("%d-%m-%Y")
	
	if not os.path.exists(outdir):
   		os.mkdir(outdir)

   	if not os.path.exists(outdir +"/"+typeOp):
   		os.mkdir(outdir +"/"+typeOp)

   	outdir=outdir +"/"+typeOp
   	
	output = pd.DataFrame( data={"id":test_data["screen_name"], "realAge":test_data["age"], "ageRandomForest":resultForest,"ageNaiveBayes":resultBayes,"ageSVM":resultSVM,"ageSGD":resultSGD})
	
	# Use pandas to write the comma-separated output file
	outname = 'tweets_customFields_results.csv'
	fullname = os.path.join(outdir, outname)    
	output.to_csv(fullname,index=False)
	
	# View a list of the features and their importance scores
	headers = ["name", "score"]
	print "Importance of Features: "#, sorted(list(zip(train_data[features], forest.feature_importances_)), key=lambda x: x[1])

	values = sorted(zip(train_data_features, forest.feature_importances_), key=lambda x: x[1] * -1)
	print tabulate(values, headers, tablefmt="plain")

	#############################################
	# EVALUATE THE MODEL
	#############################################
	print "Evaluating the model --> Calculating metrics ..."

	db_access = MongoDBUtils()

	ageRanges=[]
   	if typeOp=='normal':
   		ageRanges=db_access.getAgeRanges()
   	else:
   		ageRanges=db_access.get3AgeRanges()

   	target_names = ageRanges	
   	
   	data = df_complete[['friends_count',  'tweets_count', 'linkedin', 'snapchat', 'instagram','facebook','followers_count','favourites_count','qtyMentions','qtyHashtags','qtyUrls', 'qtyEmojis', 'qtyUppercase','profile_pic_gender']]
	y_complete = ml_utils.convertToInt(df_complete['age'],typeOp)

	#--------------
	##BAYES
	#--------------
	name_prefix='customFields_'+typeOp+'_'+balanced

	print "Metrics for Naive Bayes:"
	ml_utils.createConfusionMatrix(test_data['age'].tolist(),resultBayes,ageRanges,name_prefix,'NaiveBayes',outdir)
	print classification_report(test_data['age'].tolist(), resultBayes, target_names=target_names)

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
	print "10-Fold Accuracy: ", accuracyRF 

	#--------------
	##SVM
	#--------------
	print "Metrics for SVM:"
	ml_utils.createConfusionMatrix(test_data['age'].tolist(),resultSVM,ageRanges,name_prefix,'SVM',outdir)
	print classification_report(test_data['age'].tolist(), resultSVM, target_names=target_names)
	
	scores = cross_val_score(svm, data, y_complete, cv=StratifiedKFold(n_splits=10, shuffle=True, random_state = 5),scoring=make_scorer(accuracy_score))
	accuracySVM = round(scores.mean(),2)	
	print "10-Fold Accuracy: ", accuracySVM

	#--------------
	##SGD
	#--------------
	print "Metrics for SGD:"
	ml_utils.createConfusionMatrix(test_data['age'].tolist(),resultSGD,ageRanges,name_prefix,'SGD',outdir)
	print classification_report(test_data['age'].tolist(), resultSGD, target_names=target_names)

	scores = cross_val_score(sgd, data, y_complete, cv=StratifiedKFold(n_splits=10, shuffle=True, random_state = 5),scoring=make_scorer(accuracy_score))
	accuracySGD = round(scores.mean(),2)	
	print "10-Fold Accuracy: ", accuracySGD 

	#--------------
	##OUTPUT
	#--------------
	result= "ACCURACY--> N.Bayes:",accuracyNB,"|RForest:", accuracyRF,"|SVM:", accuracySVM,"|SGD:", accuracySGD
	print result
	return result	# Copy the results to a pandas dataframe 
	output = pd.DataFrame( data={"id":test_data["screen_name"], "realAge":test_data["age"], "ageRandomForest":resultForest,"ageNaiveBayes":resultBayes})
	

if __name__ == '__main__':
    main_customFields()
