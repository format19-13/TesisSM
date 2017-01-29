#!/usr/bin/python
# -*- coding: utf8 -*-
import logging
import logging.handlers
import os

LOGGING_MSG_FORMAT = '%(levelname)-8s %(asctime)-15s %(name)-12s %(message)s'


def init_logger(root_name="twitter_streamer", level='DEBUG', log_base_path=None):
    print 'initializing logger'
    # Crea el logger
    logger = logging.getLogger(root_name)
    # Setea el nivel de criticidad

    if level == 'CRITICAL':
        level = logging.CRITICAL
    elif level == 'ERROR':
        level = logging.ERROR
    elif level == 'WARNING':
        level = logging.WARNING
    elif level == 'INFO':
        level = logging.INFO
    elif level == 'DEBUG':
        level = logging.DEBUG
    else:
        level = logging.NOTSET
    logger.setLevel(level)

    # Crea el handler
    if log_base_path is None:
        script_path = os.path.dirname(__file__)
        path = os.path.join(script_path, 'logs', root_name)
    else:
        path = os.path.join(log_base_path, root_name)

    print 'log file in' + str(path)

    handler = logging.handlers.TimedRotatingFileHandler(path, 'midnight', 1)
    handler.setLevel(level)

    # Establece el formato y asocia el handler al logger
    formatter = logging.Formatter(LOGGING_MSG_FORMAT)
    handler.setFormatter(formatter)
    handler.suffix = "%Y-%m-%d-%H-%M"  # or anything else that strftime will allow
    logger.addHandler(handler)

