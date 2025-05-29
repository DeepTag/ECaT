# --*--coding:utf-8 --*--
# Author: meng
# Time: 2:54 AM 8/6/20 
# CBIM at the Department of Computer Science 
# Rutgers University

# ----------------------------------------------------------------------------------
# imports
import logging
from enum import Enum

# ----------------------------------------------------------------------------------
#
class LOGGER(Enum):
    STREAM = 0
    FILE = 1
    PRINT = 2
    NONE = 3


LOGGER_TYPE = LOGGER.STREAM
LOGGER_FORMAT = '%(asctime)2s [%(pathname)2s line:%(lineno)d] %(levelname)-6s %(message)s'

# ----------------------------------------------------------------------------------
#  general logging
def get_logger(logger_name='medinfer'):
    logger = logging.getLogger(logger_name)
    if LOGGER_TYPE == LOGGER.STREAM:
        steam_handler = logging.StreamHandler()
        steam_handler.setFormatter(logging.Formatter(LOGGER_FORMAT))
        logger.addHandler(steam_handler)
        logger.setLevel(logging.INFO)
    return logger