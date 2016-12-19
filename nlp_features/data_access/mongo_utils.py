#!/usr/bin/python
# -*- coding: utf8 -*-

# __author__ = 'eviotti'

from twitter_timeline_extractor.configs.data_bases import *
from twitter_timeline_extractor.configs.settings import *
from pymongo.errors import PyMongoError, ConnectionFailure
import pymongo
import logging


class MongoDBUtils(object):

    def __init__(self):
        # Cliente a MongoDB
        self.mongo_client = pymongo.MongoClient(host=MONGO_DB_HOST, port=MONGO_DB_PORT)
        self.mongo_client.admin.authenticate(MONGO_DB_USER, MONGO_DB_PASSWORD, mechanism='MONGODB-CR')
        self.logger = logging.getLogger(LOGGING_ROOT_NAME + '.data_access')
        self.logger.info('Initializing module.')

    def save_tweet(self, document, source):

        try:

            # Obtiene una referencia a la instancia de la DB
            db = self.mongo_client[MONGO_DB_NAME]

            # Obtiene el ObjectID Mongo del perfil del data source para el usuario
            col = db[DB_COL_TWEETS]

            # FLAGS indicando como se configuro el streamer que trajo el tweet(track_terms, follow o bounding box)
            document["geolocation"] = True if EnumSource.GEOLOCATION in source else False
            document["follow"] = True if EnumSource.FOLLOW in source else False
            document["track_terms"] = True if EnumSource.TRACKTERMS in source else False

            col.insert_one(document)

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
        col.update({'screen_name' : screen_name }, 
                     {'$set' : {'age' : age }})
