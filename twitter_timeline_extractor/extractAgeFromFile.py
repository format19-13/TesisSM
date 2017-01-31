import glob
import os
import xml.etree.ElementTree as XML
import string
import itertools as IT
import io

#!/usr/bin/python
# -*- coding: utf8 -*-
import os, sys
sys.path.append(os.path.abspath(os.pardir))

from configs.settings import *
from configs.data_bases import *
from data_access.mongo_utils import MongoDBUtils

from requests.exceptions import ChunkedEncodingError
from twython import Twython
from threading import Thread
from Queue import Queue
import streamer_logging
import traceback
import logging
import sys
import time
import pymongo
from pymongo import MongoClient
import imp

def set_age(screen_name,age):
    db_access = MongoDBUtils()
    db_access.set_age_user(screen_name,age)

with open(DIR_PREFIX+"/Dropbox/TesisVT/pan16-author-profiling-training-dataset-2016-04-25/pan16-author-profiling-training-dataset-spanish-2016-04-25/truth.txt") as f:
    for line in f:
        user_id = line.split(":::")[0]
        age_range=line.split(":::")[2]
        with open(DIR_PREFIX+"/Dropbox/TesisVT/pan16-author-profiling-training-dataset-2016-04-25/pan16-author-profiling-training-dataset-spanish-2016-04-25/"+user_id+".xml") as g:
        
            e = XML.parse(g).getroot()
            screen_name=string.replace(e.attrib['url'], 'https://twitter.com/', '').lower()    
            print screen_name
            print age_range
            print '------'
            set_age(screen_name,age_range)


