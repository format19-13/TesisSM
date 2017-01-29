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


class Features():
    db_access = MongoDBUtils()
    users_dicc = db_access.get_users()

    for user in list(users_dicc):
        print user
    aux=list(users_dicc)
    
    categories = ['alt.atheism', 'soc.religion.christian','comp.graphics', 'sci.med']
    twenty_train = fetch_20newsgroups(subset='train', categories=categories, shuffle=True, random_state=42)
    
    print twenty_train.data[0]

    print twenty_train.target
    print len(twenty_train.target)
    print len(twenty_train.data)
    print max(twenty_train.target)
    print min(twenty_train.target)

def main():
    print 'Process start...'
    processor = Features()
    print 'Exiting now.'

if __name__ == "__main__":
    main()