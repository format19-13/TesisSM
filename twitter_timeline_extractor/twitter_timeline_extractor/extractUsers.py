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


def save_user(data):
    """
    This is the worker thread function.
    It processes items in the queue one after
    another.  These daemon threads go into an
    infinite loop, and only exit when
    the main thread ends.
    """
    print "guardo user"
    db_access = MongoDBUtils()
    db_access.save_user(data)
    print data


class TwitterStreamer(Twython):

    def __init__(self, source):
        """

        :param source: Lista de Enumerdados que indican la fuente del tweet en la configuración del streamer.
        Esto puede ser por follow accounts, track terms o bounding box
        :return:
        """

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

        self.module_logger.debug(TRACK_TERMS)
        self.module_logger.debug(FOLLOWS_IDS)


        try:
               
            track_terms = TRACK_TERMS if len(TRACK_TERMS) > 0 else None
            follow = FOLLOWS_IDS if len(FOLLOWS_IDS) > 0 else None
            locations = BOUNDING_BOXES if len(BOUNDING_BOXES) > 0 else None
            user_id=USER_ID
            output=self.lookup_user(screen_name="sboxcoaching,ingeosolum")
			
            for user in output:
                print "usuario:"
                print user['screen_name']
                save_user(user)
			
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
