
# coding=utf-8
# This Python file uses the following encoding: utf-8
import os,sys
reload(sys)
sys.setdefaultencoding('utf8')
import os.path
import numpy as np
import pandas as pd
sys.path.append(os.path.abspath(os.pardir))
from sklearn.model_selection import train_test_split
from configs.settings import *
from data_access.mongo_utils import MongoDBUtils
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
import string
import re
import nltk
import HTMLParser
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize

# Simple WordCloud
from os import path
from scipy.misc import imread
import matplotlib.pyplot as plt
import random

from wordcloud import WordCloud

def main():
	##WORDCLOUD FOR EVERY AGE RANGE
	db_access = MongoDBUtils()
	ageRanges = db_access.getAgeRanges()
	#ageRanges=['18-24']
	htmlParser= HTMLParser.HTMLParser()

	for ar in ageRanges:

		#Decode data
		result = db_access.get_SubscriptionListsFromAgeRange(ar)
		
		wordcloud = WordCloud(width=1600, height=800).generate(result)
		# Open a plot of the generated image.
		plt.figure( figsize=(20,10), facecolor='k')
		plt.title('wordcloud subscription lists:'+ ar)
		plt.imshow(wordcloud)
		plt.axis("off")
		plt.tight_layout(pad=0)
		plt.savefig('wordcloud_subscriptions'+ ar +".png", facecolor='k', bbox_inches='tight')


if __name__ == '__main__':
    main()
