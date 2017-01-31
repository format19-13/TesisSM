#!/usr/bin/python
# -*- coding: utf8 -*-

import os,sys
import os.path

sys.path.append(os.path.abspath(os.pardir))
from configs.settings import *
from data_access.mongo_utils import MongoDBUtils

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.datasets import fetch_20newsgroups

import traceback
import logging
import sys
import time
import pymongo
from pymongo import MongoClient
import imp
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfTransformer

from time import time
import sys
import os
import numpy as np
import scipy.sparse as sp
import matplotlib.pyplot as plt

from sklearn.datasets import load_mlcomp
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.naive_bayes import MultinomialNB


class Features():

    def split_into_lemmas(tweet):
        bigram_vectorizer = CountVectorizer(ngram_range=(1, 3), token_pattern=r'\b\w+\b', min_df=1)
        analyze = bigram_vectorizer.build_analyzer()
        return analyze(tweet)

    db_access = MongoDBUtils()
    users_dicc = db_access.get_users()
    train_data= []
    train_target= []
    test_data= []
    test_target= []
    test_result= []
    cont=0

    # Load the training set
    for user in list(users_dicc):
        for tweet in user['tweets']:
            try:
                if (cont<45730):
                    train_target.append(user['age'])
                    train_data.append(tweet['text'])
                else:
                     test_data.append(tweet['text'])
                     test_result.append(user['age'])
                cont= cont+1
            except:
                print("Sin edad: " + user['screen_name']) 
    print test_result
    print("Extracting features from the dataset using a sparse vectorizer")
    t0 = time()
    vectorizer = TfidfVectorizer(encoding='latin1')
    X_train = vectorizer.fit_transform(train_data)
    print("done in %fs" % (time() - t0))
    print("n_samples: %d, n_features: %d" % X_train.shape)
    assert sp.issparse(X_train)
    y_train = train_target

    print("Extracting features from the dataset using the same vectorizer")
    t0 = time()
    X_test = vectorizer.transform(test_data)
    y_test = test_target
    print("done in %fs" % (time() - t0))
    print("n_samples: %d, n_features: %d" % X_test.shape)

    print X_test

    def benchmark(clf_class, params, name,X_train,y_train,X_test,y_test,test_target):
        print("parameters:", params)
        t0 = time()
        clf = clf_class(**params).fit(X_train, y_train)
        print("done in %fs" % (time() - t0))

        if hasattr(clf, 'coef_'):
            print("Percentage of non zeros coef: %f" % (np.mean(clf.coef_ != 0) * 100))
        print("Predicting the outcomes of the testing set")
        t0 = time()
        pred = clf.predict(X_test)
        print("done in %fs" % (time() - t0))

        print("Classification report on test set for classifier:")
        print(clf)
        #print(classification_report(y_test, pred))

        #cm = confusion_matrix(y_test, pred)
        #print("Confusion matrix:")
        #print(cm)

        # Show confusion matrix
        #plt.matshow(cm)
        #plt.title('Confusion matrix of the %s classifier' % name)
        #plt.colorbar()


    print("Testbenching a linear classifier...")
    parameters = {
        'loss': 'hinge',
        'penalty': 'l2',
        'n_iter': 50,
        'alpha': 0.00001,
        'fit_intercept': True,
    }

    benchmark(SGDClassifier, parameters, 'SGD',X_train,y_train,X_test,y_test,test_target)

    print("Testbenching a MultinomialNB classifier...")
    parameters = {'alpha': 0.01}

    benchmark(MultinomialNB, parameters, 'MultinomialNB',X_train,y_train,X_test,y_test,test_target)

    plt.show()

def main():
    print('Process start...')
    processor = Features()
    print ('Exiting now.')

if __name__ == "__main__":
    main()