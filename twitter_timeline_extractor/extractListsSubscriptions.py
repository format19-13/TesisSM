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

def save_listSubscriptions(screen_name,lists):
    db_access = MongoDBUtils()
    db_access.save_listSubscriptions(screen_name,lists)

class TwitterStreamerSubscriptions(Twython):

    def __init__(self, source):

        self.name = 'twitter_streamer'
        self.count = 0

        # Logger
        Twython.__init__(self, TWITTER_ACCESS_KEYS["app_key"], TWITTER_ACCESS_KEYS["app_secret"],
                                 TWITTER_ACCESS_KEYS["app_access_token"],
                                 TWITTER_ACCESS_KEYS["app_access_token_secret"])

    def run(self):
        db_access = MongoDBUtils()
        users = db_access.get_users("users");
        contador=1
        for user in users:
            try:
                if not self.hasSubscriptionLists(user):

                    if contador < 15:
                        contador = contador + 1
                        print '----------------------------'
                        print user["screen_name"]
                        lists=self.get_list_subscriptions(screen_name=user['screen_name'],count=1000)                
                        print len(lists["lists"])
                        save_listSubscriptions(user["screen_name"], lists["lists"])
                        contador=contador+1
                    else:
                        print "esperando"
                        time.sleep(900)
                        contador=0

            except Exception as e : 
                print 'Error subscription lists user: ',user["screen_name"]
                print e
                save_listSubscriptions(user["screen_name"], -1)


    def hasSubscriptionLists(self,user):
        try:
            user['listsSubscriptions']
            return True
        except:
            return False
            
def main():
    print 'Process start...'
    processor = TwitterStreamerSubscriptions(source=SOURCE)
    processor.run()
    print 'Exiting now.'

if __name__ == "__main__":
    main()
