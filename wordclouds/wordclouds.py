
# coding=utf-8
import os, sys
sys.path.append(os.path.abspath(os.pardir))

import numpy as np
import pandas as pd
from cloud_of_words_tweets import main_wordcloudsTweets
from cloud_of_words_following import main_wordcloudsFollowing

print "Creating wordclouds for Tweets by Age Ranges..."
main_wordcloudsTweets()


print "Creating wordclouds for Subscription Lists by Age Ranges..."
main_wordcloudsFollowing()





