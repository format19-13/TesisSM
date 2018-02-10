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
import imp
# Simple WordCloud
from os import path
from scipy.misc import imread
import matplotlib.pyplot as plt
import random
from nltk.corpus import stopwords
from wordcloud import WordCloud

def main_wordcloudsTweets():
	##WORDCLOUD FOR EVERY AGE RANGE
	db_access = MongoDBUtils()
	ageRanges = db_access.getAgeRanges()
	#ageRanges=['18-24']
	htmlParser= HTMLParser.HTMLParser()

	##STOPWORDS EN SPANISH, SCIKIT TRAE SOLO EN INGLES
	foo = imp.load_source('stopwords', DIR_PREFIX+'/proyectos/TesisVT/nlp_features/nlp_utils.py')
	stopwords = foo.generateCustomStopwords()  

	for ar in ageRanges:
		print ar
		#Decode data
		df_tweets=pd.read_csv(DATASET_PATH+"/tweets_"+ar+".csv", sep=",",dtype=str)
	
		text = ' '.join(df_tweets['tweets'])
		punct = string.punctuation.replace("#", "¿¡")
		result = remove_from_string(text, punct)
		words=result.split()

	 	textFiltered=u""
		
		#stop = set(stopwords.words('spanish'))
		
		for w in words:	
			print w
			if w not in stopwords:
				textFiltered=textFiltered +u' '+ unicode(w,"utf-8","ignore")

		wordcloud = WordCloud(width=1600, height=800).generate(textFiltered)
		print "Dibujando wordcloud para ", ar, " ..."

		# Open a plot of the generated image.
		plt.figure( figsize=(20,10), facecolor='k')
		plt.title('wordcloud ages:'+ ar)
		plt.imshow(wordcloud)
		plt.axis("off")
		plt.tight_layout(pad=0)
		plt.savefig('wordcloud_'+ ar +".png", facecolor='k', bbox_inches='tight')

def remove_from_string(string, chars_to_remove):
	""" Removes all occurances of a the given characters
	from the string. """
	result = ""
	for char in string:
		if not char in chars_to_remove:
			result = result + char

	return result 

def generateCustomStopwords():

	stopset = set(nltk.corpus.stopwords.words('spanish'))

	##add domain stopwords
	stopset.add((u'hoy'))
	stopset.add((u'más'))
	stopset.add((u'si'))
	stopset.add((u'aquí'))
	stopset.add((u'ahora'))
	stopset.add((u'está'))
	stopset.add((u'ser'))
	stopset.add((u'bien'))
	stopset.add((u'gracias'))
	stopset.add((u'qué'))
	stopset.add((u'día'))
	stopset.add((u'días'))
	stopset.add((u'año'))
	stopset.add((u'años'))
	stopset.add((u'mejor'))
	stopset.add((u'puede'))
	stopset.add((u'hacer'))
	stopset.add((u'así'))
	stopset.add((u'hace'))
	stopset.add((u'ver'))
	stopset.add((u'cómo'))
	stopset.add((u'va'))
	stopset.add((u'españa'))
	stopset.add((u'vía'))
	stopset.add((u'gran'))
	stopset.add((u'nuevo'))
	return stopset
