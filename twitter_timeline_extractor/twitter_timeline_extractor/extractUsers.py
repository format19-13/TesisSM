#!/usr/bin/python
# -*- coding: utf8 -*-
import os, sys
sys.path.append(os.path.abspath(os.pardir))

from twitter_timeline_extractor.configs.settings import *
from requests.exceptions import ChunkedEncodingError
from data_access.mongo_utils import MongoDBUtils
from twython import Twython
from threading import Thread
from Queue import Queue
import streamer_logging
import traceback
import logging
import sys
import time


def save_user(data):
    db_access = MongoDBUtils()
    db_access.save_user(data)

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
        try:

            screen_names_lst=SCREEN_NAMES.split(",")
            screenName='"'
            for x in range(1, len(screen_names_lst)+1):
                if len(screenName)==1 :
                    screenName=screenName+screen_names_lst[x-1]
                else: 
                    screenName=screenName+","+screen_names_lst[x-1]

                if (x % 50==0):
                    screenName=screenName + '"'   
                    output=self.lookup_user(screen_name=screenName)
                    print "USUARIOS GUARDADOS EN BD:"
                    for user in output:
                        print user['screen_name']
                        tweets= self.get_user_timeline(screen_name=user['screen_name'], count=3000)
                        user['tweets']=tweets
                        save_user(user)
                    screenName='"'
                    time.sleep(900)

        except ChunkedEncodingError as e:
            msg = "ChunkedEncodingError in execution of the search track processor. " + str(e)
            self.module_logger.debug(msg)
            self.module_logger.debug(traceback.format_exc())
            self.module_logger.debug("Resetting streamer...")
        except Exception as e:
            print str(e)
            self.module_logger.rollback()
            self.module_logger.close()
            msg = "Unexpected error in execution of the search track processor. " + str(e)
            self.module_logger.debug(msg)
            self.module_logger.debug(traceback.format_exc())
            self.module_logger.debug("Resetting streamer...")


def main():
    print 'Process start...'
    processor = TwitterStreamer(source=SOURCE)
    processor.run()
    print 'Exiting now.'

if __name__ == "__main__":
    main()
