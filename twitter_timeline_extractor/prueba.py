#!/usr/bin/python
# -*- coding: utf8 -*-
import os, sys
sys.path.append(os.path.abspath(os.pardir))

from configs.settings import *
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

class TwitterStreamer(Twython):

    def __init__(self, source):

        self.name = 'twitter_streamer'
        self.count = 0

        # Logger
        streamer_logging.init_logger(root_name=LOGGING_ROOT_NAME, level='DEBUG', log_base_path=LOGGING_BASE_PATH)
        self.module_logger = logging.getLogger(LOGGING_ROOT_NAME + '.streamer')
        self.module_logger.info("Starting twitter streamer...")

        Twython.__init__(self, TWITTER_ACCESS_KEYS["app_key"], TWITTER_ACCESS_KEYS["app_secret"],
                                 TWITTER_ACCESS_KEYS["app_access_token"],
                                 TWITTER_ACCESS_KEYS["app_access_token_secret"])

  
    def run(self):
        db_access = MongoDBUtils()           
        output=self.lookup_user(screen_name='michael_sorano')
        print output
                   
 

def main():
    print 'Process start...'
    processor = TwitterStreamer(source=SOURCE)
    processor.run()
    print 'Exiting now.'

if __name__ == "__main__":
    main()
