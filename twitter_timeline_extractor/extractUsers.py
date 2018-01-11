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

    def save_user(self,data):
        db_access = MongoDBUtils()
        db_access.save_user(data)

    def __init__(self, source):

        self.name = 'twitter_streamer'
        self.count = 0

        # Logger
        #streamer_logging.init_logger(root_name=LOGGING_ROOT_NAME, level='DEBUG', log_base_path=LOGGING_BASE_PATH)
        #self.module_logger = logging.getLogger(LOGGING_ROOT_NAME + '.streamer')
        #self.module_logger.info("Starting twitter streamer...")

        Twython.__init__(self, TWITTER_ACCESS_KEYS["app_key"], TWITTER_ACCESS_KEYS["app_secret"],
                                 TWITTER_ACCESS_KEYS["app_access_token"],
                                 TWITTER_ACCESS_KEYS["app_access_token_secret"])

  
    def run(self):
        db_access = MongoDBUtils()
        usersUnlabeled = db_access.get_users("unlabeled_users")
        cont=0
        screen_names = ''

        for user_unlab in usersUnlabeled:
            age = db_access.getEdad(user_unlab['screen_name'],'unlabeled_users')

            if age != -1 and not db_access.userExistsInDb(user_unlab['screen_name'].lower(),'users') :                     
                cont = cont+1
                print user_unlab['screen_name']

                if screen_names == '':
                    screen_names = user_unlab['screen_name']
                else:
                    screen_names = screen_names + ','+ user_unlab['screen_name']

                if cont == 99 : #lookup_user: 100 requests every 15 min
                    print 1
                    output=self.lookup_user(screen_name=screen_names)

                    print "USUARIOS GUARDADOS EN BD:"
                    for user in output:
                        try:
                            userToSave = self.getUserTweetsAndInfo(user)
                            userToSave["age"]=db_access.getEdad(userToSave['screen_name'],"unlabeled_users")
                            print userToSave['screen_name']
                            self.save_user(user)
                        except Exception as e: 
                            print "Error al intentar guardar usuario: ", user["screen_name"]
                            print(e)
                            pass
                        
                    screenName=''                        
                    print "esperando"
                    time.sleep(900) 
                    cont = 0
                    screen_names=""

        if len(screen_names) != 0:
            output=self.lookup_user(screen_name=screen_names)
            print "USUARIOS GUARDADOS EN BD:"
            for user in output: 
                print user['screen_name']   
                try:     
                    userToSave = self.getUserTweetsAndInfo(user)
                    userToSave["age"]=db_access.getEdad(userToSave['screen_name'],"unlabeled_users")
                    print userToSave['screen_name']
                    self.save_user(user)   
                except Exception as e: 
                    print "Error al intentar guardar usuario: ", user["screen_name"]
                    print(e)
                    pass

    def getUserTweetsAndInfo(self,user):
        tweets= self.get_user_timeline(screen_name=user['screen_name'], count=3000)
        tweetsInSpanish=[]

        for tweet in tweets:                            
            if tweet["lang"]=="es":
                tweetsInSpanish.append(tweet)
            
        user['tweets']=tweetsInSpanish                    
        user['screen_name']= user['screen_name'].lower()
        user["linkedin"] = False                                 
        user["instagram"] = False
        user["snapchat"] = False 

        try:                    
            urls= user["entities"]["url"]["urls"]
            for url in urls:                                      
                if "linkedin" in url["expanded_url"] : user["linkedin"] = True 
                if "instagram" in url["expanded_url"]: user["instagram"] = True 
                if "snapchat" in url["expanded_url"] : user["snapchat"] = True                              
        except Exception as e: 
                #"Usuario: ", user['screen_name'], "no tiene links asociados"
                pass
        return user

    def getAgeFromFacebook(self,user):
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

    def getLatestProfilePic(self,screen_name,image):
        try : 
            user=self.lookup_user(screen_name=screen_name)
            profilePic=user[0]["profile_image_url_https"]
            db_access = MongoDBUtils()
            db_access.updateProfilePicture(screen_name,profilePic)
            return profilePic.replace("normal", "400x400")
        except :
            return image  


def main():
    print 'Process start...'
    processor = TwitterStreamer(source=SOURCE)
    processor.run()
    print 'Exiting now.'

if __name__ == "__main__":
    main()
