
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
import re
import nltk
import HTMLParser
#nltk.download("stopwords")
#nltk.download('punkt')
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize

# Simple WordCloud
from os import path
from scipy.misc import imread
import matplotlib.pyplot as plt
import random

from wordcloud import WordCloud

##STOPWORDS SET

stopset = set(nltk.corpus.stopwords.words('spanish'))
stopwords=set()

##add domain stopwords
stopset.add(("https"))
stopset.add(("co"))
stopset.add(("CO"))
stopset.add(("RT"))
stopset.add(("Hoy"))
stopset.add(("más"))
stopset.add(("si"))
stopset.add(("aquí"))
stopset.add(("ahora"))
stopset.add(("está"))
stopset.add(("ser"))
stopset.add(("bien"))
stopset.add(("gracias"))
stopset.add(("Gracia"))
stopset.add(("Cómo"))
stopset.add(("qué"))
stopset.add(("Qué"))
stopset.add(("día"))
stopset.add(("es"))
stopset.add(("Gracia"))


for x in stopset:
	x=x.encode('utf-8')
	stopwords.add(x)

##WORDCLOUD FOR EVERY AGE RANGE
db_access = MongoDBUtils()
ageRanges = db_access.getAgeRanges()
htmlParser= HTMLParser.HTMLParser()

for ar in ageRanges:

#Decode data
	text = db_access.get_tweetsTextFromAgeRange(ar)
	result= htmlParser.unescape(text)
	result= unicode(result.encode("utf8"), errors='ignore')
	result= re.sub(r"http\S+", "",result)

	words = word_tokenize(result)
 	textFiltered=""

	for w in words:
		if w not in stopwords:
			textFiltered=textFiltered + w.encode("utf-8")

	print textFiltered
	wordcloud = WordCloud(width=1600, height=800).generate(textFiltered)
	# Open a plot of the generated image.
	plt.figure( figsize=(20,10), facecolor='k')
	plt.title('wordcloud ages:'+ ar)
	plt.imshow(wordcloud)
	plt.axis("off")
	plt.tight_layout(pad=0)
	plt.savefig('wordcloud_'+ ar +".png", facecolor='k', bbox_inches='tight')
