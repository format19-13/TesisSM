
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

## *********ARMO EL DATASET DE TRAIN Y EL DE TEST *********
db_access = MongoDBUtils()
users_df = db_access.get_tweetsText()

train_data=users_df.sample(frac=0.9,random_state=200)
test_data=users_df.drop(train_data.index)


print test_data

count_vect = CountVectorizer() #Para hacer bag of words
X_train_counts = count_vect.fit_transform(train_data.tweets)
# fit_transform() does two functions: First, it fits the model
# and learns the vocabulary; second, it transforms our training data
# into feature vectors. The input to fit_transform should be a list of 
# strings.

## ********* APLICO BAG OF WORDS *********

print count_vect.vocabulary_.get(u'algorithm')
print count_vect.vocabulary_.get(u'amigos')

#scikit no trae stopwords en espanol, solo en ingles.
# Numpy arrays are easy to work with, so convert the result to an 
# array
train_data_features = X_train_counts.toarray()

print len(train_data) #212 users en train

print train_data_features.shape 
#(212, 94210) --> It has 212 rows and 94210 (one for each vocabulary word).

# Take a look at the words in the vocabulary
#vocab = count_vect.get_feature_names()
#print vocab 
##Aca vemos strings raros q deberiamos eliminar ej: \u0432\u0435\u043b\u0438\u043a\u0438\u043c

# Sum up the counts of each vocabulary word
dist = np.sum(train_data_features, axis=0)

# For each, print the vocabulary word and the number of times it 
# appears in the training set
#for tag, count in zip(vocab, dist):
 #   print count, tag

# ********* APLICO RANDOM FOREST Y LO ENTRENO CON LA DATA EN TRAIN*********#

print "Training the random forest..."

# Initialize a Random Forest classifier with 100 trees
forest = RandomForestClassifier(n_estimators = 100) 
# Fit the forest to the training set, using the bag of words as 
# features and the sentiment labels as the response variable
#
# This may take a few minutes to run

print train_data["age"]

forest = forest.fit( train_data_features, train_data["age"] )

# ********* APLICO RANDOM FOREST SOBRE LA DATA DE TEST *********#

# Read the test data

# Get a bag of words for the test set, and convert to a numpy array
test_data_features = count_vect.transform(test_data.tweets)
test_data_features = test_data_features.toarray()

# Use the random forest to make sentiment label predictions
result = forest.predict(test_data_features)

# Copy the results to a pandas dataframe with an "id" column and
# a "sentiment" column
print result.str.replace('\n', '')

output = pd.DataFrame( data={"id":test_data["screen_name"], "age":result,"realAge":test_data["age"]})
#print output

# Use pandas to write the comma-separated output file
output.to_csv( "Bag_of_Words_model.csv", index=False)
