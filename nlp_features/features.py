#!/usr/bin/python
# -*- coding: utf8 -*-

import sys
sys.path.append('/home/vero/proyectos/TesisVT/twitter_streamer_extractor/twitter_streamer_extractor/data_access')
from mongo_utils import MongoDBUtils

sys.path.append('/home/vero/proyectos/TesisVT/twitter_streamer_extractor/twitter_streamer_extractor/configs')
from settings import *
from data_bases import *

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

    #for user in list(users_dicc):
     #   print user
    #aux=list(users_dicc)
    
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
