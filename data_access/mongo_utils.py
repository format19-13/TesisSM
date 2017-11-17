#!/usr/bin/python
# -*- coding: utf8 -*-

import os,sys
import os.path
from pandas import DataFrame
sys.path.append(os.path.abspath(os.pardir))
from configs.settings import *
from configs.data_bases import *

from pymongo.errors import PyMongoError, ConnectionFailure
import pymongo
import logging


class MongoDBUtils(object):

    def __init__(self):
        # Cliente a MongoDB
        self.mongo_client = pymongo.MongoClient(host=MONGO_DB_HOST, port=MONGO_DB_PORT)
        self.mongo_client.tesisdb.authenticate(MONGO_DB_USER, MONGO_DB_PASSWORD, mechanism='SCRAM-SHA-1')
        self.logger = logging.getLogger(LOGGING_ROOT_NAME + '.data_access')
        self.logger.info('Initializing module.')

    def save_tweet(self, document, source):

        try:

            # Obtiene una referencia a la instancia de la DB
            db = self.mongo_client[MONGO_DB_NAME]

            # Obtiene el ObjectID Mongo del perfil del data source para el usuario
            col = db[DB_COL_UNLABELED_TWEETS]

            # FLAGS indicando como se configuro el streamer que trajo el tweet(track_terms, follow o bounding box)
           # document["geolocation"] = True if EnumSource.GEOLOCATION in source else False
           # document["follow"] = True if EnumSource.FOLLOW in source else False
            #document["track_terms"] = True if EnumSource.TRACKTERMS in source else False

            col.insert_one(document)

        except ConnectionFailure as e:
            self.logger.error('Mongo connection error', exc_info=True)
        except PyMongoError as e:
            self.logger.error('Error while trying to save user account', exc_info=True)

    def get_users(self):

        try:

            # Obtiene una referencia a la instancia de la DB
            db = self.mongo_client[MONGO_DB_NAME]
            # Obtiene el ObjectID Mongo del perfil del data source para el usuario
            col = db[DB_COL_USERS]
            return col.find(no_cursor_timeout=True)

        except ConnectionFailure as e:
            self.logger.error('Mongo connection error', exc_info=True)
        except PyMongoError as e:
            self.logger.error('Error while trying to sace user account', exc_info=True)

    def save_user(self, document):
            db = self.mongo_client[MONGO_DB_NAME]
            col = db[DB_COL_USERS]
            col.insert_one(document)

    def set_age_user(self, screen_name,age):
            db = self.mongo_client[MONGO_DB_NAME]
            col = db[DB_COL_USERS]
            col.update({'screen_name' : screen_name }, {'$set' : {'age' : age }})

    def get_tweetsText(self):

        try:
            df = DataFrame(columns=('screen_name', 'tweets', 'age','followers_count',  'tweets_count', 'linkedin', 'snapchat', 'instagram'))
            # Obtiene una referencia a la instancia de la DB
            db = self.mongo_client[MONGO_DB_NAME]
            # Obtiene el ObjectID Mongo del perfil del data source para el usuario
            col = db[DB_COL_USERS]
            count=0
            for user in col.find(): #para cada usuario
                tweetText=""
                for tweet in  user['tweets']:
                    tweetText= tweetText +' '+ tweet['text']
                #print user['screen_name']
                #print user['age']
                df.loc[count] = [user['screen_name'],tweetText,user['age'],user['followers_count'],len(user['tweets']),user['linkedin'],user['snapchat'],user['instagram'] ]
                count += 1
            return df

        except ConnectionFailure as e:
            self.logger.error('Mongo connection error', exc_info=True)
        except PyMongoError as e:
            self.logger.error('Error while trying to save user account', exc_info=True)

    def get_tweetsTextFromAgeRange(self, ageRange):

        try:          
            # Obtiene una referencia a la instancia de la DB
            db = self.mongo_client[MONGO_DB_NAME]
            # Obtiene el ObjectID Mongo del perfil del data source para el usuario
            col = db[DB_COL_USERS]
            tweetText=""
            for user in col.find({"age":ageRange}) :
                for tweet in  user['tweets']:
                    tweetText= tweetText +' '+ tweet['text']
            return tweetText        
        except ConnectionFailure as e:
            self.logger.error('Mongo connection error', exc_info=True)
        except PyMongoError as e:
            self.logger.error('Error while trying to save user account', exc_info=True)

    def getAgeRanges(self):

        try:          
            # Obtiene una referencia a la instancia de la DB
            db = self.mongo_client[MONGO_DB_NAME]
            # Obtiene el ObjectID Mongo del perfil del data source para el usuario
            col = db[DB_COL_USERS]
            return col.distinct( "age" )   
        except ConnectionFailure as e:
            self.logger.error('Mongo connection error', exc_info=True)
        except PyMongoError as e:
            self.logger.error('Error while trying to save user account', exc_info=True)

    def get_customFields(self):

        try:
            df = DataFrame(columns=('screen_name', 'followers_count',  'tweets_count', 'linkedin', 'snapchat', 'instagram','age'))
            # Obtiene una referencia a la instancia de la DB
            db = self.mongo_client[MONGO_DB_NAME]
            # Obtiene el ObjectID Mongo del perfil del data source para el usuario
            col = db[DB_COL_USERS]
            count=0
            for user in col.find(): #para cada usuario
                tweetText=""
                df.loc[count] = [user['screen_name'],user['followers_count'],len(user['tweets']),user['linkedin'],user['snapchat'],user['instagram'],user['age'] ]
                count += 1
            return df

        except ConnectionFailure as e:
            self.logger.error('Mongo connection error', exc_info=True)
        except PyMongoError as e:
            self.logger.error('Error while trying to save user account', exc_info=True)

    def save_listSubscriptions(self, screen_name,lists):
        db = self.mongo_client[MONGO_DB_NAME]
        col = db[DB_COL_USERS]
        col.update({'screen_name' : screen_name }, {'$set' : {'listsSubscriptions' : lists }})