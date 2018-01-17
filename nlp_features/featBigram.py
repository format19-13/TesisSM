
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
from nltk.corpus import stopwords
from tabulate import tabulate
from sklearn.feature_extraction.text import CountVectorizer

## *********ARMO EL DATASET DE TRAIN Y EL DE TEST *********
db_access = MongoDBUtils()
users_df = db_access.get_tweetsTextForBigrams()

for (i,row) in users_df["tweets"].iteritems():
	result=''
	result=re.sub(' RT ',"", row)
	result= re.sub(r"http\S+", "",result)
	result=re.sub(r'@\w+',"", result)
	users_df["tweets"][i]=result.encode("utf-8").decode("utf-8")

# Split into training and test set
# 80% of the input for training and 20% for testing

train_data=users_df.sample(frac=0.8,random_state=200) 
test_data=users_df.drop(train_data.index)

bigram_vectorizer = CountVectorizer(ngram_range=(2,3), token_pattern=r'\b\w+\b', min_df=1,strip_accents='unicode',max_features=500) 

X_train_counts = bigram_vectorizer.fit_transform(train_data.tweets)
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
vocab = bigram_vectorizer.get_feature_names()
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
test_data_features = bigram_vectorizer.transform(test_data.tweets)
test_data_features = test_data_features.toarray()

# Use the random forest to make age range predictions
resultForest = forest.predict(test_data_features)

resultBayes = bayes.predict(test_data_features)

# Copy the results to a pandas dataframe with an "id" column and
# a "age" column

output = pd.DataFrame( data={"id":test_data["screen_name"], "realAge":test_data["age"], "ageRandomForest":resultForest,"ageNaiveBayes":resultBayes})
#print output

# Use pandas to write the comma-separated output file
output.to_csv( "Bigram_model_ForestAndBayes.csv", index=False)

# View a list of the features and their importance scores
#print "Importance of Features: ", sort(zip(train_data_features, forest.feature_importances_))

headers = ["name", "score"]
values = sorted(zip(vocab, forest.feature_importances_), key=lambda x: x[1] * -1)
print(tabulate(values, headers, tablefmt="plain"))

###################################
#******* MODEL EVALUATION *********
###################################

#create confusion matrix: anything on the diagonal was classified correctly and the rest incorrectly.
cnf_matrix =confusion_matrix(test_data['age'].tolist(), resultForest)
print "Confusion Matrix for Random Forest: "
print cnf_matrix

cnf_matrix2 =confusion_matrix(test_data['age'].tolist(), resultBayes)
print "Confusion Matrix for Naive Bayes: "
print cnf_matrix2

