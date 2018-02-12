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
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
import imp
import sys
# Simple WordCloud
from os import path
from scipy.misc import imread
import matplotlib.pyplot as plt
import random
from nltk.corpus import stopwords
from wordcloud import WordCloud
from nlp_features.customStopwords import getCustomStopwords,getSpanishStopwords

def anyPresent(text, stopWords):
	for word in stopWords:
		if (' '+word+' ') in text:
			return True

	return False

def removeStopWords(text, stopwords):
	originaltext = text

	while anyPresent(text, stopwords):
		for word in stopwords:
			text = (str(text)).replace(' ' + str(word) + ' ', ' ')

	return text

def main_wordcloudsTweets():
	##WORDCLOUD FOR EVERY AGE RANGE
	db_access = MongoDBUtils()
	ageRanges = db_access.getAgeRanges()
	#ageRanges=['50-64']
	stopwords = getCustomStopwords()  
	stopwords.append(u'jajaja')
	stopwords.append(u'gracia')
	stopwords.append(u'asi')
	stopwords.append(u'via')
	stopwords.append(u'dia')
	stopwords.append(u'tambien')
	stopsAux=[]
	for stop in stopwords:
		stopsAux.append(stop.encode('utf-8'))

	for ar in ageRanges:
		print ar
		#Decode data
		df_tweets=pd.read_csv(DATASET_PATH+"/tweets_"+ar+".csv", sep=",")
		
		text=''
		for tw in df_tweets['tweets']:
			tw=tw.translate(None, string.punctuation)
			tw=tw.replace('¿', ' ')
			tw=tw.replace('¡', ' ')
			tw=tw.replace('á', 'a')
			tw=tw.replace('é', 'e')
			tw=tw.replace('í', 'i')
			tw=tw.replace('ó', 'o')
			tw=tw.replace('ú', 'u')
			# Replace all stop words from the tweet
			text += removeStopWords(tw, stopwords)

		text = removeStopWords(text, stopwords)
	
		wordcloud = WordCloud(width=1600, height=800).generate(text)
		print "Dibujando wordcloud para ", ar, " ..."

		# Open a plot of the generated image.
		plt.figure( figsize=(20,10), facecolor='k')
		plt.title('wordcloud ages:'+ ar)
		plt.imshow(wordcloud)
		plt.axis("off")
		plt.tight_layout(pad=0)
		plt.savefig('wordcloud_'+ ar +".png", facecolor='k', bbox_inches='tight')