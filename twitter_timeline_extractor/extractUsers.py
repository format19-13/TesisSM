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
            screenName=''

            for x in range(0, len(screen_names_lst)):

                if len(screenName)==1 :
                    screenName=screenName+screen_names_lst[x]
                else: 
                    screenName=screenName+","+screen_names_lst[x]

                if ((x % 50==0 and x != 0) or (x+1==len(screen_names_lst))):

                    output=self.lookup_user(screen_name=screenName)

                    print "USUARIOS GUARDADOS EN BD:"
                    for user in output:
                        print user['screen_name']
                        tweets= self.get_user_timeline(screen_name=user['screen_name'], count=3000)
                        user['tweets']=tweets
                        #user['ageRange']= 
                        #self.getAgeFromFacebook(user)
                        ageRange=""
                        user["linkedin"] = False 
                        user["instagram"] = False
                        user["snapchat"] = False 
                        try:
                            urls= user["entities"]["url"]["urls"]
                            for url in urls:
                                if "facebook" in url["expanded_url"]:
                                    fbk_url= url["expanded_url"]
                                    fbk_username=fbk_url.split("facebook.com")[1].split("/")[1]
                                    foo = imp.load_source('scrapingFacebook', DIR_PREFIX+'/proyectos/TesisVT/facebook_extractor/scrapingFacebook.py')
                                    ageRange= foo.getEdad(fbk_username)
                                    user["age"]=ageRange
                                
                                if "linkedin" in url["expanded_url"] : user["linkedin"] = True 
                                if "instagram" in url["expanded_url"]: user["instagram"] = True 
                                if "snapchat" in url["expanded_url"] : user["snapchat"] = True 
                               
                        except:
                            pass ##no tiene facebook acct asociada

                        save_user(user)
                    screenName=''
                    if len(screen_names_lst)>=50:
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
    
    def getAgeFromFacebook(user):
        ageRange=""
        try:
            urls= user["entities"]["url"]["urls"]
            for url in urls:
                if "facebook" in url["expanded_url"]:
                    fbk_url= url["expanded_url"]
                    fbk_username=fbk_url.split("facebook.com")[1].split("/")[1]
                    print fbk_username
                    foo = imp.load_source('scrapingFacebook', DIR_PREFIX+'/proyectos/TesisVT/facebook_extractor/scrapingFacebook.py')
                    ageRange= foo.getEdad(fbk_username)
        except:
            pass ##no tiene facebook acct asociada

        return ageRange
 

def main():
    print 'Process start...'
    processor = TwitterStreamer(source=SOURCE)
    processor.run()
    print 'Exiting now.'

if __name__ == "__main__":
    main()
