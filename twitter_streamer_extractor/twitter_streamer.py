#!/usr/bin/python
# -*- coding: utf8 -*-
import os, sys
sys.path.append(os.path.abspath(os.pardir))

from configs.settings import *
from configs.data_bases import *
from data_access.mongo_utils import MongoDBUtils

from requests.exceptions import ChunkedEncodingError
from twython import TwythonStreamer
from threading import Thread
from Queue import Queue
import streamer_logging
import traceback
import logging
import sys


def process_twitter_data(worker_id, queue, module_name, source):
    """
    This is the worker thread function.
    It processes items in the queue one after
    another.  These daemon threads go into an
    infinite loop, and only exit when
    the main thread ends.
    """

    logger = logging.getLogger(LOGGING_ROOT_NAME + '.processor.' + str(worker_id))
    logger.debug("Worker" + str(worker_id) + " looking for data...")

    db_access = MongoDBUtils()

    while True:
        data = queue.get()
        if 'text' in data:
            # Guarda el Tweet
            db_access.save_tweet(data, source)
            print data

            logger.debug('TWEET | id: ' + str(data['id']) + ': ' + data['text'].encode('utf-8'))
        elif 'delete' in data:
            logger.debug('DELETION NOTICE | ' + str(data).encode('utf-8'))
        elif 'warning' in data:
            logger.debug('STALL WARNING | ' + str(data).encode('utf-8'))
        elif 'limit' in data:
            logger.debug('LIMIT NOTICE | ' + str(data).encode('utf-8'))
        elif 'disconnect' in data:
            logger.debug('DISCONNECTION MESSAGE | ' + str(data).encode('utf-8'))
        elif 'status_withheld' in data:
            logger.debug('STATUS WITHHELD | ' + str(data).encode('utf-8'))
        elif 'user_withheld' in data:
            logger.debug('USER WITHHELD | ' + str(data).encode('utf-8'))
        else:
            logger.debug('PRETTY ODD | Data: ' + str(data))

        queue.task_done()


class TwitterStreamer(TwythonStreamer):

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

        TwythonStreamer.__init__(self, TWITTER_ACCESS_KEYS["app_key"], TWITTER_ACCESS_KEYS["app_secret"],
                                 TWITTER_ACCESS_KEYS["app_access_token"],
                                 TWITTER_ACCESS_KEYS["app_access_token_secret"])
        self.queue = Queue()
        self.workers = []
        for i in range(CANT_WORKERS):
            worker = Thread(target=process_twitter_data, args=(i, self.queue, self.name, source))
            worker.setDaemon(True)
            worker.start()
            self.workers.append(worker)

    # STREAMER EVENTS
    def on_success(self, data):
        self.count += 1
        self.queue.put(data)
    
    def on_error(self, status_code, data):
        msg = "Unexpected error in execution of the Twitter streamer process (status_code: "+str(status_code)+"). " \
              + str(data)
        self.module_logger.debug(msg)
        self.module_logger.debug(traceback.format_exc())

        if int(status_code) == 420:
            self.module_logger.debug(msg)
            self.module_logger.debug("Exiting now...")
            sys.exit(0)
    # END STREAMER EVENTS
  
    def run(self):

        self.module_logger.debug(TRACK_TERMS)
        self.module_logger.debug(FOLLOWS_IDS)

        while True:
            try:
                track_terms = TRACK_TERMS if len(TRACK_TERMS) > 0 else None
                follow = FOLLOWS_IDS if len(FOLLOWS_IDS) > 0 else None
                locations = BOUNDING_BOXES if len(BOUNDING_BOXES) > 0 else None
                self.statuses.filter(track=track_terms, follow=follow, stall_warnings=True,
                                     locations=locations)
            except ChunkedEncodingError as e:
                msg = "ChunkedEncodingError in execution of the search track processor. " + str(e)
                self.module_logger.debug(msg)
                self.module_logger.debug(traceback.format_exc())
                self.module_logger.debug("Resetting streamer...")
            except Exception as e:
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
