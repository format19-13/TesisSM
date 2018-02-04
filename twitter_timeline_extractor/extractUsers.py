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
import emoji
import re

class TwitterStreamer(Twython):

    def save_user(self,data):
        db_access = MongoDBUtils()
        db_access.save_user(data)

    def markUnlabeledAsLabeled(self,userUnlabeled):
        db_access = MongoDBUtils()
        db_access.markUnlabeledAsLabeled(userUnlabeled['screen_name'])

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
        usersUnlabeled = db_access.get_unlabeled_users_with_age()
        cont=0
        screen_names = ''

        for user_unlab in usersUnlabeled:
            age = user_unlab['ageRange']

            if  not db_access.userExistsInDb(user_unlab['screen_name'].lower(),'users') :                     
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
                        print user['screen_name']   
                        try:     
                            try:
                                userToSave = self.getUserTweets(user)
                            except Exception as e:
                                print "Usuario con perfil restringido"
                                userToSave = user

                            userToSave = self.populateOtherNetworks(userToSave)
                            userToSave = self.populate_mentions_hashtags_urls(userToSave)

                            userToSave["age"]=db_access.getEdad(userToSave['screen_name'],"unlabeled_users")
                            userToSave["exactAge"]=db_access.getExactAge(userToSave['screen_name'])
                    
                            print userToSave['screen_name']
                            
                            try:
                                self.save_user(userToSave)
                                self.markUnlabeledAsLabeled(userToSave)
                            except pymongo.errors.DocumentTooLarge as e:
                                while True:
                                    print "********* Doc muy grande, eliminando 50 tweets..."
                                    try: 
                                        userToSave['tweets']= userToSave['tweets'][:len(userToSave['tweets'])-50]
                                        self.save_user(userToSave)
                                        self.markUnlabeledAsLabeled(userToSave)
                                        break
                                    except pymongo.errors.DocumentTooLarge as e:
                                        pass

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
                    try:
                        userToSave = self.getUserTweets(user)
                    except Exception as e:
                        print "Usuario con perfil restringido"
                        userToSave = user

                    userToSave = self.populateOtherNetworks(userToSave)
                    userToSave = self.populate_mentions_hashtags_urls(userToSave)
                    userToSave["age"]=db_access.getEdad(userToSave['screen_name'],"unlabeled_users")
                    userToSave["exactAge"]=db_access.getExactAge(userToSave['screen_name'])
                    
                    print userToSave['screen_name']
                    userToSave['screen_name']= user['screen_name'].lower()
                    
                    try:
                        self.save_user(userToSave)
                        self.markUnlabeledAsLabeled(userToSave)
                    except pymongo.errors.DocumentTooLarge as e:
                        while True:                                
                            print "********* Doc muy grande, eliminando 50 tweets..."
                            try: 
                                userToSave['tweets']= userToSave['tweets'][:len(userToSave['tweets'])-50]
                                userToSave = self.populate_mentions_hashtags_urls(userToSave)
                                self.save_user(userToSave)
                                self.markUnlabeledAsLabeled(userToSave)
                                break
                            except pymongo.errors.DocumentTooLarge as e:
                                pass

                except Exception as e: 
                            print "Error al intentar guardar usuario: ", user["screen_name"]
                            print(e)
                            pass

    def getUserTweets(self,user):
        try:
            alltweets = []  
            print user['screen_name'], "-Getting tweets"
            #make initial request for most recent tweets (200 is the maximum allowed count)
            new_tweets = self.get_user_timeline(screen_name = user['screen_name'],count=200, tweet_mode='extended')

            alltweets.extend(new_tweets)
        
            #save the id of the oldest tweet less one
            oldest = alltweets[-1]['id'] - 1

            #keep grabbing tweets until there are no tweets left to grab
            while len(new_tweets) > 0:
                #print "getting tweets before %s" % (oldest)
                
                #all subsiquent requests use the max_id param to prevent duplicates
                try : 
                    new_tweets = self.get_user_timeline(screen_name = user['screen_name'],count=200, tweet_mode='extended',max_id=oldest)
                
                except Exception as e:
                    print e
                    print "esperando"
                    time.sleep(900) 
                    new_tweets = self.get_user_timeline(screen_name = user['screen_name'],count=200, tweet_mode='extended',max_id=oldest)
                
                #save most recent tweets
                alltweets.extend(new_tweets)

                #update the id of the oldest tweet less one
                oldest = alltweets[-1]['id'] - 1

                print "...%s tweets downloaded so far" % (len(alltweets))

            tweetsSpanish=[]
            for tweet in alltweets:
                if tweet["lang"]=="es" and not (tweet['full_text'].startswith('RT ')):
                    tweetsSpanish.append(tweet)

            print "from ", len(alltweets), ' tweets, ',len(tweetsSpanish), 'were in Spanish and not RT'
            
            user['tweets']=tweetsSpanish  

        except Exception as e:
            print e

        return user

    def populateOtherNetworks(self,user):    
        print user['screen_name'], "- populating other networks"       

        user["linkedin"] = False                                 
        user["instagram"] = False
        user["snapchat"] = False 
        user["facebook"] = False

        try: 
            #First try to get them from urls                   
            urls= user["entities"]["url"]["urls"]
            for url in urls:                                      
                if "linkedin" in url["expanded_url"] : user["linkedin"] = True 
                if "instagram" in url["expanded_url"]: user["instagram"] = True 
                if "snapchat" in url["expanded_url"] : user["snapchat"] = True     
                if "facebook" in url["expanded_url"] : user["facebook"] = True                           
        except Exception as e: 
                #"Usuario: ", user['screen_name'], "no tiene links asociados"
                pass

        #Next try to get them from the bio
        bio = user['description']

        if user['lang'] == 'es' and isinstance(bio, unicode):
            bio = bio.lower()
            screen_name = user["screen_name"]
                
            if (bio.find(u'instagram')!= -1 or bio.find(u'ig:')!= -1 or bio.find(u'insta')!= -1) and (not user["instagram"]) : 
                user["instagram"] = True 
            if (bio.find(u'snap')!= -1 or bio.find(u'snapchat:')!= -1) and (not user["snapchat"]): 
                user["snapchat"] = True     
            if bio.find(u'linkedin')!= -1 and (not user["linkedin"]): 
                user["linkedin"] = True 
            if (bio.find(u'facebook')!= -1 or bio.find(u'face:')!= -1 or bio.find(u'fbk')!= -1) and (not user["facebook"]) : 
                user["facebook"] = True 

        if user["facebook"]: print "Usuario: ", screen_name, " tiene facebook"
        if user["instagram"]: print "Usuario: ", screen_name, " tiene instagram"
        if user["snapchat"]: print "Usuario: ", screen_name, " tiene snapchat"
        if user["linkedin"]: print "Usuario: ", screen_name, " tiene linkedin"
        
        return user

    def getAgeFromFacebook(self,user):
        print user['screen_name'], "- getting age from facebook"     
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
        print screen_name, "- getting latest profile pic"     
        try : 
            user=self.lookup_user(screen_name=screen_name)
            profilePic=user[0]["profile_image_url_https"]
            db_access = MongoDBUtils()
            db_access.updateProfilePicture(screen_name,profilePic)
            return profilePic.replace("normal", "400x400")
        except :
            return image  

    def updateTweets(self):
        db_access = MongoDBUtils()
        cont=0
        for user in db_access.get_users("users"):
            cont = cont+1
            if (len(user['tweets'])==0):
                print cont
                oldTweets= len(user['tweets'])
                try:
                    userToSave = self.getUserTweets(user)
                except Exception as e:
                    print "Usuario con perfil restringido"
                    userToSave= populate_mentions_hashtags_urls(user)
                #print "ANTES:",oldTweets , "- AHORA: ", len(userToSave['tweets'])

                if oldTweets != len(userToSave['tweets']):
                    try:
                        db_access.save_user_tweets(user["screen_name"], userToSave['tweets'])
                    except pymongo.errors.DocumentTooLarge as e:
                        while True:
                            print "********* Doc muy grande, eliminando 50 tweets..."
                            try: 
                                userToSave['tweets']= userToSave['tweets'][:len(userToSave['tweets'])-50]
                                db_access.save_user_tweets(user["screen_name"], userToSave['tweets'])
                                break
                            except pymongo.errors.DocumentTooLarge as e:
                                pass
    def updateOtherNetworks(self):
        db_access = MongoDBUtils()
        cont=0
        for user in db_access.get_users("users"):
            cont = cont+1
            if (cont>0):
                print cont

                try:
                    userToSave = self.populateOtherNetworks(user)
                    db_access.save_other_network(userToSave['screen_name'],'facebook',userToSave['facebook'])
                except Exception as e:
                    print "error"

    def populate_mentions_hashtags_urls(self,user):
        print user['screen_name'], "- populate_mentions_hashtags_urls"     
        qtyMentions=0
        qtyHashtags=0
        qtyUrls=0
        qtyEmojis=0

        emojis_list = map(lambda x: ''.join(x.split()), emoji.UNICODE_EMOJI.keys())
        r = re.compile('|'.join(re.escape(p) for p in emojis_list)) #emojis
        re2="[>]?[:;Xx][']?[-=]?[)(PpDdOo$]|<3" #emoticons

        for tweet in  user['tweets']:
            txt=tweet['full_text']

            qtyMentions=qtyMentions+len(tweet['entities']['user_mentions'])
            qtyHashtags=qtyHashtags+len(tweet['entities']['hashtags'])
            qtyUrls=qtyUrls+len(tweet['entities']['urls'])
            qtyEmojis= qtyEmojis + len(r.findall(txt))+len(re.findall(re2,txt))

            qtyTweets = len(user['tweets'])

            user["qtyMentions"]=round(qtyMentions/qtyTweets,2)
            user["qtyHashtags"]=round(qtyHashtags/qtyTweets,2)
            user["qtyUrls"]=round(qtyUrls/qtyTweets,2)
            user["qtyEmojis"]=round(qtyEmojis/qtyTweets,2)
        return user
                   
def main():
    print 'Process start...'
    processor = TwitterStreamer(source=SOURCE)
    processor.run()
    print 'Exiting now.'

if __name__ == "__main__":
    main()
