
import os,sys
import os.path
import numpy as np
import pandas as pd
sys.path.append(os.path.abspath(os.pardir))
from sklearn.model_selection import train_test_split
from configs.settings import *
from data_access.mongo_utils import MongoDBUtils
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier

# Simple WordCloud
from os import path
from scipy.misc import imread
import matplotlib.pyplot as plt
import random

from stop_words import get_stop_words

from wordcloud import WordCloud, STOPWORDS

##STOPWORDS SET

stopwordsSP = (get_stop_words('spanish'))

stopwords=set(STOPWORDS)

for x in stopwordsSP:
	x=x.encode('utf-8')
	stopwords.add(x)

##add domain stopwords
stopwords.add("https")
stopwords.add("co")
stopwords.add("RT")

##WORDCLOUD FOR EVERY AGE RANGE
db_access = MongoDBUtils()
ageRanges = db_access.getAgeRanges()

for ar in ageRanges:
	text = db_access.get_tweetsTextFromAgeRange(ar)
	wordcloud = WordCloud(stopwords = stopwords,width=1600, height=800).generate(text)

	# Open a plot of the generated image.
	plt.figure( figsize=(20,10), facecolor='k')
	plt.title('wordcloud ages:'+ ar)
	plt.imshow(wordcloud)
	plt.axis("off")
	plt.tight_layout(pad=0)
	plt.savefig('wordcloud_'+ ar +".png", facecolor='k', bbox_inches='tight')
