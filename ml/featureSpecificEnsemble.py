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
		if allVotes[0][index] != allVotes[1][index] and allVotes[0][index] != allVotes[2][index] and allVotes[1][index] != allVotes[2][index]:
			ret.append(allVotes[0][index])
		else:
			tmp_list = [allVotes[0][index],allVotes[1][index],allVotes[2][index]]
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
	if balanced == 'balanced':
		train_data_sub=pd.read_csv(DATASET_PATH+"/subscriptionLists_balanced_train.csv", sep=",",dtype=str)
		test_data_sub=pd.read_csv(DATASET_PATH+"/subscriptionLists_balanced_test.csv", sep=",",dtype=str)
	else:
		train_data_sub=pd.read_csv(DATASET_PATH+"/"+typeOp+"_subscriptionLists_train.csv", sep=",",dtype=str)
		test_data_sub=pd.read_csv(DATASET_PATH+"/"+typeOp+"_subscriptionLists_test.csv", sep=",",dtype=str)
	final_train_data = pd.merge(train_data, train_data_sub, how='left', on=['screen_name','age'])
	final_train_data = final_train_data.fillna('')
	print final_train_data
	final_test_data = pd.merge(test_data, test_data_sub, how='left', on=['screen_name','age'])
	final_test_data = final_test_data.fillna('')
	frames = [final_train_data, final_test_data]
	df_complete= pd.concat(frames)
	train_data_sgd=final_train_data[['screen_name','tweets','age']]
	test_data_sgd=final_test_data[['screen_name','tweets','age']]
	train_data_custom_fields=final_train_data[['screen_name','friends_count',  'tweets_count', 'linkedin', 'snapchat', 'instagram','facebook','followers_count','favourites_count','qtyMentions','qtyHashtags','qtyUrls', 'qtyEmojis', 'qtyUppercase','profile_pic_gender','age']]
	test_data_custom_fields=final_test_data[['screen_name','friends_count',  'tweets_count', 'linkedin', 'snapchat', 'instagram','facebook','followers_count','favourites_count','qtyMentions','qtyHashtags','qtyUrls', 'qtyEmojis', 'qtyUppercase','profile_pic_gender','age']]

	print 'Number of observations in the training data:', len(final_train_data)
	print 'Number of observations in the test data:',len(final_test_data)
	frames = [train_data_sgd, test_data_sgd]
	df_complete= pd.concat(frames)
	print 'Number of observations in the whole dataset:',len(df_complete)
	#forest
	forest = RandomForestClassifier(n_estimators=140, max_depth=20,min_samples_leaf=2 )
	features_custom_fields_forest = train_data_custom_fields.columns[1:(len(test_data_custom_fields.columns)-1)]
	train_data_features_for_custom_fields_forest=train_data_custom_fields[features_custom_fields_forest]
	#sgd
	sgd = SGDClassifier(loss='log', penalty='l2', random_state=42, alpha=0.0001,n_iter=60)
	stopwords = getCustomStopwords()  
	transformer_tfidf = TfidfVectorizer(smooth_idf=False,lowercase=False,stop_words=stopwords,max_features=5000, ngram_range=(1,3))
	tfidf = transformer_tfidf.fit_transform(final_train_data.tweets)
	train_data_features_ngrams_sgd = tfidf.toarray()
	#svm
	svm = SVC(kernel='rbf', C= 0.08, gamma= 0.1)
	count_vect = CountVectorizer(stop_words=stopwords, max_features=5000,ngram_range=(1,3), token_pattern=r'\b\w+\b' )
	X_train_counts = count_vect.fit_transform(final_train_data.subscriptionLists)
	train_data_features_sub = X_train_counts.toarray()
	#print ml_utils.SVM_param_selection(train_data_features_sub, final_train_data["age"]) {'kernel': 'rbf', 'C': 8, 'gamma': 1}
	#ensemble
	print "train ensemble"
	features = [train_data_features_for_custom_fields_forest, train_data_features_ngrams_sgd, train_data_features_sub]
	estimators = fit_multiple_estimators([forest, sgd, svm], features, final_train_data['age'], None)
	print "finish training ensemble"
	#test

	#sgd
	transformer_tfidf = TfidfVectorizer(smooth_idf=False,lowercase=False,stop_words=stopwords,max_features=5000, ngram_range=(1,3))
	tfidf = transformer_tfidf.fit_transform(test_data_sgd.tweets)
	test_data_features_sgd = tfidf.toarray()
	#svm
	
	test_data_features_sub = count_vect.transform(final_test_data.subscriptionLists)
	test_data_features_sub = test_data_features_sub.toarray()
	#forest
	test_data_features_for_custom_fields_forest=test_data_custom_fields[features_custom_fields_forest]
	#ensamble
	featues_test = [test_data_features_for_custom_fields_forest,test_data_features_sgd, test_data_features_sub]
	y_pred = predictClass(estimators, featues_test)
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
	








