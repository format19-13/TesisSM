
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

## *********ARMO EL DATASET DE TRAIN Y EL DE TEST *********
db_access = MongoDBUtils()
users_df = db_access.get_tweetsText()


##https://www.analyticsvidhya.com/blog/2015/06/quick-guide-text-data-cleaning-python/
##TWEETS CLEANUP
#https://github.com/myleott/ark-twokenize-py
#Decode data

for (i,row) in users_df["tweets"].iteritems():
	result=''
	result=re.sub(' RT ',"", row)
	result= re.sub(r"http\S+", "",result)
	result=re.sub(r'@\w+',"", result)
	users_df["tweets"][i]=result

# Split into training and test set
# 80% of the input for training and 20% for testing

train_data=users_df.sample(frac=0.8,random_state=200) 
test_data=users_df.drop(train_data.index)

##STOPWORDS EN SPANISH, SCIKIT TRAE SOLO EN INGLES
foo = imp.load_source('stopwords', DIR_PREFIX+'/proyectos/TesisVT/nlp_features/nlp_utils.py')
stopwords = foo.generateCustomStopwords()  

count_vect = CountVectorizer(stop_words=stopwords, max_features=500 ) #Para hacer bag of words
X_train_counts = count_vect.fit_transform(train_data.tweets)
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

# Initialize a Random Forest classifier with 100 trees
forest = MultinomialNB()
# Fit the forest to the training set, using the bag of words as 
# features and the age range as the response variable

forest = forest.fit( train_data_features, train_data["age"] ) 

# ********* APLICO RANDOM FOREST SOBRE LA DATA DE TEST *********#

# Read the test data

# Get a bag of words for the test set, and convert to a numpy array
test_data_features = count_vect.transform(test_data.tweets)
test_data_features = test_data_features.toarray()

# Use the random forest to make age range predictions
result = forest.predict(test_data_features)

# Copy the results to a pandas dataframe with an "id" column and
# a "age" column

output = pd.DataFrame( data={"id":test_data["screen_name"], "age":result,"realAge":test_data["age"]})
#print output

# Use pandas to write the comma-separated output file
output.to_csv( "Bag_of_Words_model.csv", index=False)

###################################
#******* MODEL EVALUATION *********
###################################

print 'BOW+RandomForest: ',accuracy_score(result.tolist(),test_data["age"].values.tolist()),' accuracy'

clf = RandomForestClassifier(n_estimators=100) #verificar con otros valores
# 10-Fold Cross validation
print np.mean(cross_val_score(clf, train_data_features, train_data["age"]))

target_names=list(set(train_data["age"].values.tolist()))

print confusion_matrix(result.tolist(),test_data["age"].values.tolist(),target_names)

print(classification_report(result.tolist(),test_data["age"].values.tolist(), target_names=target_names))

